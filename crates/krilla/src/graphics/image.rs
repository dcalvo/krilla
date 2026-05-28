//! Creating and using bitmap images.
//!
//! krilla allows you to add bitmap images to your PDF very easily.
//! The currently supported formats include
//! - PNG
//! - JPG
//! - GIF
//! - WEBP
//! - Custom image formats via [`CustomImage`]

use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::io::Cursor;
use std::ops::DerefMut;
use std::sync::Arc;

use pdf_writer::{Array, Dict, Finish, Name, Null, Ref, Str};
use png::{BitDepth, ColorType, Transformations};
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::JpegDecoder;

use crate::chunk_container::ChunkContainer;
use crate::configure::ValidationError;
use crate::error::KrillaError;
use crate::graphics::color::{cmyk, luma, rgb};
use crate::graphics::color::{DEVICE_CMYK, DEVICE_GRAY, DEVICE_RGB};
use crate::graphics::icc::{GenericICCProfile, ICCBasedColorSpace, ICCProfile};
use crate::serialize::SerializeContext;
use crate::stream::{deflate_encode, FilterStreamBuilder};
use crate::util::{set_colorspace, Deferred, NameExt, SipHashable};
use crate::Data;

/// The number of bits per color component.
#[derive(Debug, Hash, Eq, PartialEq, Copy, Clone)]
pub enum BitsPerComponent {
    /// One bit per component.
    One,
    /// Two bits per component.
    Two,
    /// Four bits per component.
    Four,
    /// Eight bits per component.
    Eight,
    /// Sixteen bits per component.
    Sixteen,
}

impl BitsPerComponent {
    fn as_u8(&self) -> u8 {
        match self {
            BitsPerComponent::One => 1,
            BitsPerComponent::Two => 2,
            BitsPerComponent::Four => 4,
            BitsPerComponent::Eight => 8,
            BitsPerComponent::Sixteen => 16,
        }
    }
}

/// The color space of the image.
#[derive(Debug, Hash, Eq, PartialEq, Copy, Clone)]
pub enum ImageColorspace {
    /// The RGB color space.
    Rgb,
    /// The luma color space.
    Luma,
    /// The CMYK color space.
    Cmyk,
}

impl ImageColorspace {
    fn num_components(&self) -> u8 {
        match self {
            ImageColorspace::Luma => 1,
            ImageColorspace::Rgb => 3,
            ImageColorspace::Cmyk => 4,
        }
    }

    fn matches_icc_profile(&self, profile: &GenericICCProfile) -> bool {
        match self {
            ImageColorspace::Rgb => matches!(profile, GenericICCProfile::Rgb(_)),
            ImageColorspace::Luma => matches!(profile, GenericICCProfile::Luma(_)),
            ImageColorspace::Cmyk => matches!(profile, GenericICCProfile::Cmyk(_)),
        }
    }
}

impl TryFrom<ColorSpace> for ImageColorspace {
    type Error = ();

    fn try_from(value: ColorSpace) -> Result<Self, Self::Error> {
        match value {
            ColorSpace::RGB => Ok(ImageColorspace::Rgb),
            ColorSpace::RGBA => Ok(ImageColorspace::Rgb),
            ColorSpace::YCbCr => Ok(ImageColorspace::Rgb),
            ColorSpace::Luma => Ok(ImageColorspace::Luma),
            ColorSpace::LumaA => Ok(ImageColorspace::Luma),
            ColorSpace::YCCK => Ok(ImageColorspace::Cmyk),
            ColorSpace::CMYK => Ok(ImageColorspace::Cmyk),
            _ => Err(()),
        }
    }
}

struct SampledRepr {
    color_channel: Vec<u8>,
    alpha_channel: Option<Vec<u8>>,
    bits_per_component: BitsPerComponent,
}

struct JpegRepr {
    data: Data,
    bits_per_component: BitsPerComponent,
    invert_cmyk: bool,
}

/// A PDF stream filter, identified by its `/Filter` name, used by
/// [`Image::from_precompressed`] to declare how image data has already been encoded.
///
/// Distinct from the crate-private `StreamFilter` enum used by krilla's own encoders:
/// these variants name filters whose decoding krilla does not implement, but whose
/// names can be written verbatim into the `/Filter` entry of an image XObject when
/// the data is passed through from another PDF.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum PdfFilter {
    /// `FlateDecode` (DEFLATE / zlib).
    Flate,
    /// `DCTDecode` (JPEG).
    Dct,
    /// `LZWDecode`.
    Lzw,
    /// `RunLengthDecode`.
    RunLength,
    /// `CCITTFaxDecode`.
    CcittFax,
    /// `JBIG2Decode`.
    Jbig2,
    /// `JPXDecode` (JPEG 2000).
    Jpx,
    /// `ASCIIHexDecode`.
    AsciiHex,
    /// `ASCII85Decode`.
    Ascii85,
}

impl PdfFilter {
    /// The PDF name as written in a `/Filter` entry.
    pub fn to_name(self) -> Name<'static> {
        match self {
            PdfFilter::Flate => Name(b"FlateDecode"),
            PdfFilter::Dct => Name(b"DCTDecode"),
            PdfFilter::Lzw => Name(b"LZWDecode"),
            PdfFilter::RunLength => Name(b"RunLengthDecode"),
            PdfFilter::CcittFax => Name(b"CCITTFaxDecode"),
            PdfFilter::Jbig2 => Name(b"JBIG2Decode"),
            PdfFilter::Jpx => Name(b"JPXDecode"),
            PdfFilter::AsciiHex => Name(b"ASCIIHexDecode"),
            PdfFilter::Ascii85 => Name(b"ASCII85Decode"),
        }
    }
}

/// A single scalar entry of a `/DecodeParms` dictionary. PDF filter parameters
/// (CCITTFax `/K`/`/Columns`/`/BlackIs1`…, predictor `/Predictor`/`/Colors`…) are
/// all integers or booleans, so these two cover every reproducible case. Filters
/// whose parameters reference other objects (e.g. JBIG2 `/JBIG2Globals`) are not
/// representable here and must not use this passthrough.
#[derive(Debug, Clone)]
pub enum DecodeParm {
    /// An integer parameter.
    Int(i32),
    /// A boolean parameter.
    Bool(bool),
}

/// A grayscale soft mask for a precompressed image, with its own filter chain and dimensions.
///
/// PDF masks can have a different filter chain (e.g. `FlateDecode`) than the color
/// channel (e.g. `DCTDecode` for JPEG) and may also have different dimensions.
/// Emitted as a `/SMask` (DeviceGray) entry on the image XObject.
#[derive(Debug, Clone)]
pub struct AlphaChannel {
    /// Encoded alpha channel bytes (already compressed with `filters`).
    pub data: Vec<u8>,
    /// PDF stream filters in decoding order, applied to the alpha channel.
    pub filters: Vec<PdfFilter>,
    /// Per-filter `/DecodeParms`, parallel to `filters`: entry `i` holds filter
    /// `i`'s params (empty = no params for that filter). Written verbatim.
    pub decode_parms: Vec<Vec<(String, DecodeParm)>>,
    /// The `/Matte` color (in the parent image's color space) for premultiplied
    /// alpha. `None` when the alpha is straight (non-premultiplied).
    pub matte: Option<Vec<f32>>,
    /// Source `/Decode` array (per-component sample remap, e.g. `[1 0]`) for the
    /// alpha samples, written verbatim. `None` = no remap.
    pub decode: Option<Vec<f32>>,
    /// Width of the alpha channel in pixels (may differ from the color channel).
    pub width: u32,
    /// Height of the alpha channel in pixels (may differ from the color channel).
    pub height: u32,
    /// Bits per component of the alpha channel.
    pub bits_per_component: BitsPerComponent,
}

/// A 1-bit stencil mask for a precompressed image.
///
/// Emitted as a `/Mask` entry referencing a child `/ImageMask` XObject (so the
/// original 1-bit data — e.g. a JBIG2-encoded scan layer — is preserved verbatim
/// rather than decoded into a soft mask). Bits-per-component is implicitly 1.
#[derive(Debug, Clone)]
pub struct StencilMask {
    /// Encoded stencil bytes (already compressed with `filters`).
    pub data: Vec<u8>,
    /// PDF stream filters in decoding order, applied to the stencil.
    pub filters: Vec<PdfFilter>,
    /// Per-filter `/DecodeParms`, parallel to `filters` (e.g. CCITTFax
    /// `/Columns`/`/K`), empty where a filter takes none. Written verbatim.
    pub decode_parms: Vec<Vec<(String, DecodeParm)>>,
    /// Width of the stencil in pixels (may differ from the color channel).
    pub width: u32,
    /// Height of the stencil in pixels (may differ from the color channel).
    pub height: u32,
    /// When true, emit `/Decode [1 0]` to invert the stencil sense.
    pub invert: bool,
}

/// The mask accompanying a precompressed image.
#[derive(Debug, Clone)]
pub enum PrecompressedMask {
    /// Grayscale soft mask, written as `/SMask`.
    Soft(AlphaChannel),
    /// 1-bit stencil mask, written as `/Mask` referencing an `/ImageMask` XObject.
    Stencil(StencilMask),
}

/// An `/Indexed` (palette) color space for a precompressed image: the samples are
/// indices into `lookup`, resolved against a device `base` space. Emitted verbatim
/// as `[/Indexed base hival (lookup)]` so palette images pass through encoded
/// instead of being decoded to RGBA. The base is restricted to device spaces.
#[derive(Debug, Clone)]
pub struct IndexedColorSpace {
    /// Device color space the palette entries are expressed in.
    pub base: ImageColorspace,
    /// Highest valid index (palette has `hival + 1` entries).
    pub hival: u8,
    /// Palette bytes: `(hival + 1) * base.num_components()` color samples.
    pub lookup: Vec<u8>,
}

/// Image data that was already encoded by the source PDF and is being
/// passed through verbatim with its original PDF filter chain.
struct PrecompressedRepr {
    color_channel: Vec<u8>,
    mask: Option<PrecompressedMask>,
    bits_per_component: BitsPerComponent,
    /// PDF stream filters in decoding order, applied to the color channel.
    filters: Vec<PdfFilter>,
    /// Per-filter `/DecodeParms` for the color `filters`, parallel to `filters`.
    decode_parms: Vec<Vec<(String, DecodeParm)>>,
    /// When set, the samples are palette indices; emit `[/Indexed ...]` instead
    /// of `base`'s color space.
    indexed: Option<IndexedColorSpace>,
    /// Source `/Decode` array (per-component sample remap, e.g. `[1 0 ...]`),
    /// written verbatim to the image XObject's `/Decode` entry. `None` = no remap.
    color_decode: Option<Vec<f32>>,
}

enum Repr {
    Sampled(SampledRepr),
    Jpeg(JpegRepr),
    // Boxed: `PrecompressedRepr` holds several `Vec`/`Option` channels and is much
    // larger than the other variants.
    Precompressed(Box<PrecompressedRepr>),
}

impl Repr {
    fn bits_per_component(&self) -> BitsPerComponent {
        match self {
            Repr::Sampled(s) => s.bits_per_component,
            Repr::Jpeg(j) => j.bits_per_component,
            Repr::Precompressed(p) => p.bits_per_component,
        }
    }
}

/// A trait for custom images, which you can use if the
/// current methods provided by krilla (JPEG/PNG/WEBP/GIF) images
/// are not suitable for your own purpose.
///
/// Note that a struct implementing this trait should be cheap to
/// hash and clone, otherwise performance might be bad!
pub trait CustomImage: Hash + Clone + Send + Sync + 'static {
    /// Return the raw bytes of the color channel.
    fn color_channel(&self) -> &[u8];
    /// Return the raw bytes of the alpha channel, if available.
    fn alpha_channel(&self) -> Option<&[u8]>;
    /// Return the bits per component of the image.
    fn bits_per_component(&self) -> BitsPerComponent;
    /// Return the dimensions of the image.
    fn size(&self) -> (u32, u32);
    /// Return the ICC profile of the image, if available.
    fn icc_profile(&self) -> Option<&[u8]>;
    /// Return the color space of the image.
    fn color_space(&self) -> ImageColorspace;
}

struct ImageMetadata {
    size: (u32, u32),
    color_space: ImageColorspace,
    has_alpha: bool,
    bits_per_component: BitsPerComponent,
    icc: Option<GenericICCProfile>,
}

struct ImageRepr {
    inner: Deferred<Result<Repr, String>>,
    metadata: ImageMetadata,
    sip: u128,
    interpolate: bool,
}

impl ImageRepr {
    fn size(&self) -> (u32, u32) {
        self.metadata.size
    }

    fn icc(&self) -> Option<GenericICCProfile> {
        self.metadata.icc.clone()
    }

    fn color_space(&self) -> ImageColorspace {
        self.metadata.color_space
    }
}

impl Debug for ImageRepr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ImageRepr {{..}}")
    }
}

impl Hash for ImageRepr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.sip.hash(state);
    }
}

impl PartialEq for ImageRepr {
    fn eq(&self, other: &Self) -> bool {
        self.sip == other.sip
    }
}

impl Eq for ImageRepr {}

/// A bitmap image.
///
/// This type is cheap to hash and clone, but expensive to create.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct Image(Arc<ImageRepr>);

impl Image {
    /// Create a new bitmap image from a `.png` file.
    pub fn from_png(data: Data, interpolate: bool) -> Result<Image, String> {
        let hash = data.as_ref().sip_hash();
        let metadata = png_metadata(data.as_ref())?;

        Ok(Self(Arc::new(ImageRepr {
            inner: Deferred::new(move || decode_png(data.as_ref())),
            metadata,
            sip: hash,
            interpolate,
        })))
    }

    /// Create a new bitmap image from a `.jpg` file.
    pub fn from_jpeg(data: Data, interpolate: bool) -> Result<Image, String> {
        let hash = data.as_ref().sip_hash();
        let metadata = jpeg_metadata(data.as_ref())?;

        Ok(Self(Arc::new(ImageRepr {
            inner: Deferred::new(move || decode_jpeg(data)),
            metadata,
            sip: hash,
            interpolate,
        })))
    }

    /// Create a new bitmap image from a `.jpg` file with custom ICC profile.
    #[doc(hidden)]
    pub fn from_jpeg_with_icc(
        data: Data,
        icc_profile: Option<Data>,
        interpolate: bool,
    ) -> Result<Image, String> {
        let hash = data.as_ref().sip_hash();
        let mut metadata = jpeg_metadata(data.as_ref())?;
        let icc_profile =
            icc_profile.and_then(|d| get_icc_profile_type(d.as_ref(), metadata.color_space));
        metadata.icc = icc_profile;

        Ok(Self(Arc::new(ImageRepr {
            inner: Deferred::new(move || decode_jpeg(data)),
            metadata,
            sip: hash,
            interpolate,
        })))
    }

    /// Create a new bitmap image from a `.gif` file.
    pub fn from_gif(data: Data, interpolate: bool) -> Result<Image, String> {
        let hash = data.as_ref().sip_hash();
        let metadata = gif_metadata(data.as_ref())?;

        Ok(Self(Arc::new(ImageRepr {
            inner: Deferred::new(move || decode_gif(data)),
            metadata,
            sip: hash,
            interpolate,
        })))
    }

    /// Create a new bitmap image from a `.webp` file.
    ///
    /// Returns `None` if krilla was unable to parse the file.
    pub fn from_webp(data: Data, interpolate: bool) -> Result<Image, String> {
        let hash = data.as_ref().sip_hash();
        let metadata = webp_metadata(data.as_ref())?;

        Ok(Self(Arc::new(ImageRepr {
            inner: Deferred::new(move || decode_webp(data)),
            metadata,
            sip: hash,
            interpolate,
        })))
    }

    /// Create a new image from a custom image.
    ///
    /// Panics if the dimensions of the image and the length of the
    /// data doesn't match.
    pub fn from_custom<T: CustomImage>(image: T, interpolate: bool) -> Result<Image, String> {
        let hash = (image.clone(), interpolate).sip_hash();
        let metadata = ImageMetadata {
            size: image.size(),
            color_space: image.color_space(),
            has_alpha: image.alpha_channel().is_some(),
            bits_per_component: image.bits_per_component(),
            icc: image
                .icc_profile()
                .and_then(|d| get_icc_profile_type(d, image.color_space())),
        };

        Ok(Self(Arc::new(ImageRepr {
            inner: Deferred::new(move || {
                let bytes_per_component = (image.bits_per_component().as_u8() / 8) as u32;
                let color_channel_len = bytes_per_component
                    * image.color_space().num_components() as u32
                    * metadata.size.0
                    * metadata.size.1;
                let color_channel = image.color_channel();
                assert_eq!(color_channel.len(), color_channel_len as usize);

                let alpha_channel_len = bytes_per_component * metadata.size.0 * metadata.size.1;
                let alpha_channel = image.alpha_channel();
                if let Some(alpha_channel) = alpha_channel {
                    assert_eq!(alpha_channel.len(), alpha_channel_len as usize);
                }

                Ok(Repr::Sampled(SampledRepr {
                    color_channel: deflate_encode(color_channel),
                    alpha_channel: image.alpha_channel().map(deflate_encode),
                    bits_per_component: image.bits_per_component(),
                }))
            }),
            metadata,
            sip: hash,
            interpolate,
        })))
    }

    /// Create a new RGB image from raw RGBA pixels.
    pub fn from_rgba8(data: Vec<u8>, width: u32, height: u32) -> Self {
        let hash = data.sip_hash();
        let metadata = ImageMetadata {
            has_alpha: true,
            size: (width, height),
            bits_per_component: BitsPerComponent::Eight,
            color_space: ImageColorspace::Rgb,
            icc: None,
        };

        Self(Arc::new(ImageRepr {
            inner: Deferred::new(move || {
                let (color_channel, alpha_channel, bits_per_component) =
                    handle_u8_image(&data, ColorSpace::RGBA);

                Ok(Repr::Sampled(SampledRepr {
                    color_channel,
                    alpha_channel,
                    bits_per_component,
                }))
            }),
            metadata,
            sip: hash,
            interpolate: false,
        }))
    }

    /// Create an image from pre-compressed data with the given PDF stream filters.
    ///
    /// `color_channel` must already be encoded with `filters` in decoding order; the
    /// filter names are written verbatim into the resulting image XObject's `/Filter`
    /// entry. When `mask` is provided, it is emitted as either a soft `/SMask` or a
    /// 1-bit `/Mask` stencil (with its own `data`/`filters`/dimensions, independent
    /// from the color channel). No additional compression is applied to any channel.
    ///
    /// Use this to pass through image data extracted from a source PDF without round-tripping
    /// it through krilla's encoders — see [`PdfFilter`] for the supported filter names.
    ///
    /// `icc_profile`, when set, is the raw bytes of an `/ICCBased` profile whose
    /// component count matches `color_space`; it is re-embedded so color management
    /// survives the round-trip (otherwise the output declares a bare device space).
    ///
    /// `indexed`, when set, declares the samples as palette indices; the output gets
    /// an `[/Indexed ...]` color space and `color_space` is the palette's base.
    ///
    /// `color_decode`, when set, is the source `/Decode` array (a per-component sample
    /// remap such as `[1 0 ...]` inversion); it is written verbatim so the samples are
    /// interpreted as they were in the source.
    #[allow(clippy::too_many_arguments)]
    pub fn from_precompressed(
        color_channel: Vec<u8>,
        mask: Option<PrecompressedMask>,
        filters: Vec<PdfFilter>,
        decode_parms: Vec<Vec<(String, DecodeParm)>>,
        width: u32,
        height: u32,
        bits_per_component: BitsPerComponent,
        color_space: ImageColorspace,
        icc_profile: Option<Vec<u8>>,
        indexed: Option<IndexedColorSpace>,
        color_decode: Option<Vec<f32>>,
        interpolate: bool,
    ) -> Self {
        let has_alpha = mask.is_some();
        let hash = color_channel.sip_hash();
        // An `/ICCBased` source color space: re-embed the profile so the output
        // declares `[/ICCBased ...]` instead of a bare device space. Dropped if it
        // doesn't match the declared `color_space`'s component count.
        let icc = icc_profile.and_then(|d| get_icc_profile_type(&d, color_space));
        let metadata = ImageMetadata {
            size: (width, height),
            color_space,
            has_alpha,
            bits_per_component,
            icc,
        };

        Self(Arc::new(ImageRepr {
            inner: Deferred::new(move || {
                Ok(Repr::Precompressed(Box::new(PrecompressedRepr {
                    color_channel,
                    mask,
                    bits_per_component,
                    filters,
                    decode_parms,
                    indexed,
                    color_decode,
                })))
            }),
            metadata,
            sip: hash,
            interpolate,
        }))
    }

    /// Return the size of the image.
    pub fn size(&self) -> (u32, u32) {
        self.0.size()
    }

    fn icc(&self) -> Option<GenericICCProfile> {
        self.0.icc()
    }

    fn color_space(&self) -> ImageColorspace {
        self.0.color_space()
    }

    pub(crate) fn serialize(
        self,
        sc: &mut SerializeContext,
        chunk_container: &mut ChunkContainer,
        root_ref: Ref,
    ) {
        let soft_mask_id = self.0.metadata.has_alpha.then(|| {
            sc.register_validation_error(ValidationError::Transparency(sc.location));
            sc.new_ref()
        });

        let icc_ref = self.icc().and_then(|ic| {
            if sc
                .serialize_settings()
                .pdf_version()
                .supports_icc(ic.metadata())
                && self.color_space().matches_icc_profile(&ic)
            {
                let ref_ = match ic {
                    GenericICCProfile::Luma(l) => {
                        sc.register_cacheable(chunk_container, ICCBasedColorSpace(l))
                    }
                    GenericICCProfile::Rgb(r) => {
                        sc.register_cacheable(chunk_container, ICCBasedColorSpace(r))
                    }
                    GenericICCProfile::Cmyk(c) => {
                        sc.register_cacheable(chunk_container, ICCBasedColorSpace(c))
                    }
                };

                Some(ref_)
            } else {
                // Don't embed ICC profiles from images if the current
                // PDF version does not support it.
                None
            }
        });

        if self.0.interpolate {
            sc.register_validation_error(ValidationError::ImageInterpolation(sc.location));
        }

        let serialize_settings = sc.serialize_settings().clone();

        let cs = {
            let cs = match self.color_space() {
                ImageColorspace::Rgb => {
                    rgb::color_space(sc.serialize_settings().no_device_cs).into()
                }
                ImageColorspace::Luma => {
                    luma::color_space(sc.serialize_settings().no_device_cs).into()
                }
                ImageColorspace::Cmyk => match cmyk::color_space(&sc.serialize_settings()) {
                    None => {
                        sc.register_validation_error(ValidationError::MissingCMYKProfile);
                        crate::color::DeviceColorSpace::Cmyk.into()
                    }
                    Some(cs) => cs.into(),
                },
            };

            sc.register_colorspace(chunk_container, cs)
        };

        let supports_bit_depth = sc
            .serialize_settings()
            .configuration
            .version()
            .supports_bit_depth(self.0.metadata.bits_per_component);
        let location = sc.location;
        let mut chunk = sc.new_chunk();

        let chunk = Deferred::new(move || {
            if !supports_bit_depth {
                return Err(KrillaError::SixteenBitImage(self.clone(), location));
            }

            let repr = self
                .0
                .inner
                .wait()
                .as_ref()
                .map_err(|e| KrillaError::Image(self.clone(), location, e.clone()))?;

            // Set when the mask is a 1-bit stencil, so it is attached via `/Mask`
            // (referencing an `/ImageMask` XObject) rather than `/SMask`.
            let mut attach_stencil = false;

            let alpha_mask = match repr {
                Repr::Sampled(sampled) => sampled.alpha_channel.as_ref().map(|mask_data| {
                    let soft_mask_id = soft_mask_id.unwrap();
                    let mask_stream = FilterStreamBuilder::new_from_deflated(mask_data)
                        .finish(&serialize_settings);
                    let mut s_mask = chunk.image_xobject(soft_mask_id, mask_stream.encoded_data());
                    mask_stream.write_filters(s_mask.deref_mut().deref_mut());
                    s_mask.width(self.size().0 as i32);
                    s_mask.height(self.size().1 as i32);
                    s_mask.pair(
                        Name(b"ColorSpace"),
                        // Mask color space must be device gray -- see Table 145.
                        DEVICE_GRAY.to_pdf_name(),
                    );

                    if self.0.interpolate {
                        s_mask.interpolate(true);
                    }

                    s_mask.bits_per_component(repr.bits_per_component().as_u8() as i32);
                    soft_mask_id
                }),
                Repr::Precompressed(p) => p.mask.as_ref().map(|mask| {
                    let mask_id = soft_mask_id.unwrap();
                    match mask {
                        PrecompressedMask::Soft(alpha) => {
                            let mut s_mask = chunk.image_xobject(mask_id, &alpha.data);
                            write_pdf_filters(s_mask.deref_mut().deref_mut(), &alpha.filters);
                            write_decode_parms(s_mask.deref_mut().deref_mut(), &alpha.decode_parms);
                            s_mask.width(alpha.width as i32);
                            s_mask.height(alpha.height as i32);
                            s_mask.pair(Name(b"ColorSpace"), DEVICE_GRAY.to_pdf_name());

                            if self.0.interpolate {
                                s_mask.interpolate(true);
                            }

                            s_mask.bits_per_component(alpha.bits_per_component.as_u8() as i32);
                            if let Some(matte) = &alpha.matte {
                                s_mask.matte(matte.iter().copied());
                            }
                            if let Some(decode) = &alpha.decode {
                                s_mask.decode(decode.iter().copied());
                            }
                        }
                        PrecompressedMask::Stencil(stencil) => {
                            // 1-bit stencil: an `/ImageMask` XObject (no `/ColorSpace`
                            // or `/BitsPerComponent`, both implicitly fixed), referenced
                            // from the parent via `/Mask`.
                            attach_stencil = true;
                            let mut img_mask = chunk.image_xobject(mask_id, &stencil.data);
                            write_pdf_filters(img_mask.deref_mut().deref_mut(), &stencil.filters);
                            write_decode_parms(
                                img_mask.deref_mut().deref_mut(),
                                &stencil.decode_parms,
                            );
                            img_mask.width(stencil.width as i32);
                            img_mask.height(stencil.height as i32);
                            img_mask.image_mask(true);
                            if stencil.invert {
                                img_mask.decode([1.0, 0.0]);
                            }

                            if self.0.interpolate {
                                img_mask.interpolate(true);
                            }
                        }
                    }
                    mask_id
                }),
                Repr::Jpeg(_) => None,
            };

            // For Sampled/Jpeg we go through FilterStreamBuilder so the encoded data and
            // filter list are paired. Precompressed bypasses it: bytes are already encoded
            // by the source PDF and the filter names are written verbatim.
            enum ImageStream<'a> {
                Built(crate::stream::FilterStream<'a>),
                Raw {
                    bytes: &'a [u8],
                    filters: &'a [PdfFilter],
                },
            }
            let image_stream = match repr {
                Repr::Sampled(s) => ImageStream::Built(
                    FilterStreamBuilder::new_from_deflated(&s.color_channel)
                        .finish(&serialize_settings),
                ),
                Repr::Jpeg(j) => ImageStream::Built(
                    FilterStreamBuilder::new_from_jpeg_data(j.data.as_ref())
                        .finish(&serialize_settings),
                ),
                Repr::Precompressed(p) => ImageStream::Raw {
                    bytes: &p.color_channel,
                    filters: &p.filters,
                },
            };

            let encoded = match &image_stream {
                ImageStream::Built(fs) => fs.encoded_data(),
                ImageStream::Raw { bytes, .. } => bytes,
            };
            let mut image_x_object = chunk.image_xobject(root_ref, encoded);
            match &image_stream {
                ImageStream::Built(fs) => fs.write_filters(image_x_object.deref_mut().deref_mut()),
                ImageStream::Raw { filters, .. } => {
                    write_pdf_filters(image_x_object.deref_mut().deref_mut(), filters);
                    if let Repr::Precompressed(p) = repr {
                        write_decode_parms(image_x_object.deref_mut().deref_mut(), &p.decode_parms);
                    }
                }
            }
            image_x_object.width(self.size().0 as i32);
            image_x_object.height(self.size().1 as i32);

            let indexed = match repr {
                Repr::Precompressed(p) => p.indexed.as_ref(),
                _ => None,
            };
            if let Some(idx) = indexed {
                // `[/Indexed base hival (lookup)]` — samples are palette indices.
                let base = match idx.base {
                    ImageColorspace::Rgb => DEVICE_RGB,
                    ImageColorspace::Luma => DEVICE_GRAY,
                    ImageColorspace::Cmyk => DEVICE_CMYK,
                };
                let mut array = image_x_object.insert(Name(b"ColorSpace")).array();
                array.item(Name(b"Indexed"));
                array.item(Name(base.as_bytes()));
                array.item(idx.hival as i32);
                array.item(Str(&idx.lookup));
                array.finish();
            } else if let Some(icc_ref) = icc_ref {
                image_x_object.pair(Name(b"ColorSpace"), icc_ref);
            } else {
                set_colorspace(cs, image_x_object.deref_mut());
            }

            if self.0.interpolate {
                image_x_object.interpolate(true);
            }

            // Photoshop CMYK images need to be inverted, see
            // https://github.com/sile-typesetter/libtexpdf/blob/1891bee5e0b73165e4a259f910d3ea3fe1df0b42/jpegimage.c#L25-L51
            // I'm not sure if this applies to all JPEG CMYK images out there, but for now we just
            // always do it. In libtexpdf, they only seem to do it if they can find the Adobe APP
            // marker.
            if let Repr::Jpeg(j) = repr {
                if j.invert_cmyk {
                    image_x_object
                        .decode([1.0, 0.0].repeat(self.color_space().num_components() as usize));
                }
            }

            // A source color `/Decode` array passed through verbatim. Only
            // Precompressed carries one — our passthrough JPEGs are Precompressed,
            // so this also covers a JPEG's explicit `/Decode` (the Jpeg-invert above
            // is the krilla-encoded path).
            if let Repr::Precompressed(p) = repr {
                if let Some(decode) = &p.color_decode {
                    image_x_object.decode(decode.iter().copied());
                }
            }

            image_x_object.bits_per_component(repr.bits_per_component().as_u8() as i32);
            if let Some(mask_id) = alpha_mask {
                if attach_stencil {
                    image_x_object.stencil_mask(mask_id);
                } else {
                    image_x_object.s_mask(mask_id);
                }
            }
            image_x_object.finish();

            Ok(chunk)
        });

        chunk_container.streams.images.push(chunk);
    }
}

const PNG_TRANSFORMATIONS: Transformations = Transformations::EXPAND;

fn png_metadata(data: &[u8]) -> Result<ImageMetadata, String> {
    let cursor = Cursor::new(data);
    let mut decoder = png::Decoder::new(cursor);
    decoder.set_transformations(PNG_TRANSFORMATIONS);
    let reader = decoder
        .read_info()
        .map_err(|e| e.to_string().to_ascii_lowercase())?;
    let info = reader.info();

    let size = (info.width, info.height);
    let (color_type, bit_depth) = reader.output_color_type();
    let bits_per_component = match bit_depth {
        // We will normalize to 8 when decoding.
        BitDepth::One | BitDepth::Two | BitDepth::Four | BitDepth::Eight => BitsPerComponent::Eight,
        BitDepth::Sixteen => BitsPerComponent::Sixteen,
    };

    let (image_color_space, has_alpha) = match color_type {
        ColorType::Grayscale => (ImageColorspace::Luma, false),
        ColorType::GrayscaleAlpha => (ImageColorspace::Luma, true),
        ColorType::Rgb => (ImageColorspace::Rgb, false),
        ColorType::Rgba => (ImageColorspace::Rgb, true),
        ColorType::Indexed => {
            return Err("image uses an indexed color space, which is unsupported".to_string())
        }
    };
    let icc = info
        .icc_profile
        .as_ref()
        .and_then(|i| get_icc_profile_type(i, image_color_space));

    Ok(ImageMetadata {
        has_alpha,
        size,
        bits_per_component,
        color_space: image_color_space,
        icc,
    })
}

/// Write a `/Filter` entry to an image XObject dict.
///
/// Filters are emitted in the order provided (i.e. PDF decoding order). A 0-length list
/// is a no-op; a single filter is written as a name object; multiple filters are written
/// as an array.
fn write_pdf_filters(dict: &mut Dict<'_>, filters: &[PdfFilter]) {
    match filters {
        [] => {}
        [single] => {
            dict.pair(Name(b"Filter"), single.to_name());
        }
        multi => {
            dict.insert(Name(b"Filter"))
                .start::<Array>()
                .items(multi.iter().map(|f| f.to_name()));
        }
    }
}

/// Write one `/DecodeParms` dict's scalar (name, value) pairs into `dict`.
fn write_parms_dict(dict: &mut Dict<'_>, parms: &[(String, DecodeParm)]) {
    for (key, value) in parms {
        match value {
            DecodeParm::Int(i) => {
                dict.pair(Name(key.as_bytes()), *i);
            }
            DecodeParm::Bool(b) => {
                dict.pair(Name(key.as_bytes()), *b);
            }
        }
    }
}

/// Write `/DecodeParms` for a (possibly multi-) filter chain. `per_filter` is
/// parallel to the `/Filter` array — entry `i` holds filter `i`'s params, empty
/// where a filter takes none. Emits nothing when there are no params, a single
/// dict for a one-filter chain, or a `null`-padded array for a multi-filter chain
/// (so a reader binds `/DecodeParms[i]` to `/Filter[i]`).
fn write_decode_parms(dict: &mut Dict<'_>, per_filter: &[Vec<(String, DecodeParm)>]) {
    if per_filter.iter().all(|p| p.is_empty()) {
        return;
    }
    if per_filter.len() <= 1 {
        let mut parms_dict = dict.insert(Name(b"DecodeParms")).dict();
        if let Some(parms) = per_filter.first() {
            write_parms_dict(&mut parms_dict, parms);
        }
        return;
    }
    let mut array = dict.insert(Name(b"DecodeParms")).array();
    for parms in per_filter {
        if parms.is_empty() {
            array.push().primitive(Null);
        } else {
            let mut parms_dict = array.push().dict();
            write_parms_dict(&mut parms_dict, parms);
        }
    }
}

fn decode_png(data: &[u8]) -> Result<Repr, String> {
    let cursor = Cursor::new(data);
    let mut decoder = png::Decoder::new(cursor);
    decoder.set_transformations(PNG_TRANSFORMATIONS);
    let mut reader = decoder
        .read_info()
        .map_err(|e| e.to_string().to_ascii_lowercase())?;
    let mut img_data = vec![0; reader.output_buffer_size().ok_or("image is too large")?];
    let _ = reader
        .next_frame(&mut img_data)
        .map_err(|e| e.to_string())?;
    let (color_type, bit_depth) = reader.output_color_type();

    let color_space = match color_type {
        ColorType::Rgb => ColorSpace::RGB,
        ColorType::Rgba => ColorSpace::RGBA,
        ColorType::Grayscale => ColorSpace::Luma,
        ColorType::GrayscaleAlpha => ColorSpace::LumaA,
        _ => unreachable!(),
    };

    let (color_channel, alpha_channel, bits_per_component) = match bit_depth {
        BitDepth::Eight => handle_u8_image(&img_data, color_space),
        BitDepth::Sixteen => handle_u16_image(&img_data, color_space),
        _ => return Err("image has an unsupported bit-depth".to_string()),
    };

    Ok(Repr::Sampled(SampledRepr {
        color_channel,
        alpha_channel,
        bits_per_component,
    }))
}

fn jpeg_metadata(data: &[u8]) -> Result<ImageMetadata, String> {
    let reader = Cursor::new(data);
    let mut decoder = JpegDecoder::new(reader);
    decoder
        .decode_headers()
        .map_err(|e| e.to_string().to_ascii_lowercase())?;

    let size = {
        let dimensions = decoder
            .dimensions()
            .ok_or("failed to read image dimensions".to_string())?;
        (dimensions.0 as u32, dimensions.1 as u32)
    };

    let image_color_space = decoder
        .input_colorspace()
        .and_then(|c| c.try_into().ok())
        .ok_or("failed to read image colorspace".to_string())?;

    let icc = decoder
        .icc_profile()
        .and_then(|d| get_icc_profile_type(&d, image_color_space));

    Ok(ImageMetadata {
        has_alpha: false,
        size,
        bits_per_component: BitsPerComponent::Eight,
        color_space: image_color_space,
        icc,
    })
}

fn decode_jpeg(data: Data) -> Result<Repr, String> {
    let reader = Cursor::new(data.as_ref());
    let mut decoder = JpegDecoder::new(reader);
    decoder
        .decode_headers()
        .map_err(|e| e.to_string().to_ascii_lowercase())?;

    let input_color_space = decoder
        .input_colorspace()
        .ok_or("failed to read image colorspace".to_string())?;

    if matches!(
        input_color_space,
        ColorSpace::Luma
            | ColorSpace::YCbCr
            | ColorSpace::RGB
            | ColorSpace::CMYK
            | ColorSpace::YCCK
    ) {
        Ok(Repr::Jpeg(JpegRepr {
            data,
            bits_per_component: BitsPerComponent::Eight,
            invert_cmyk: matches!(input_color_space, ColorSpace::YCCK | ColorSpace::CMYK),
        }))
    } else {
        // JPEGs shouldn't be able to have a different color space?
        Err("image has an unknown color space".to_string())
    }
}

fn decode_gif(data: Data) -> Result<Repr, String> {
    let mut decoder = gif::DecodeOptions::new();
    decoder.set_color_output(gif::ColorOutput::RGBA);
    let mut decoder = decoder
        .read_info(data.as_ref())
        .map_err(|e| e.to_string().to_ascii_lowercase())?;
    let first_frame = decoder
        .read_next_frame()
        .map_err(|e| e.to_string())?
        .ok_or("GIF image seems to be empty".to_string())?;

    let (color_channel, alpha_channel, bits_per_component) =
        handle_u8_image(first_frame.buffer.as_ref(), ColorSpace::RGBA);

    Ok(Repr::Sampled(SampledRepr {
        color_channel,
        alpha_channel,
        bits_per_component,
    }))
}

fn gif_metadata(data: &[u8]) -> Result<ImageMetadata, String> {
    let size = imagesize::blob_size(data).map_err(|e| e.to_string().to_ascii_lowercase())?;

    Ok(ImageMetadata {
        // We always decode GIFs using RGBA, see `decode_gif`
        has_alpha: true,
        bits_per_component: BitsPerComponent::Eight,
        size: (size.width as u32, size.height as u32),
        color_space: ImageColorspace::Rgb,
        icc: None,
    })
}

fn webp_metadata(data: &[u8]) -> Result<ImageMetadata, String> {
    let mut decoder = image_webp::WebPDecoder::new(std::io::Cursor::new(data))
        .map_err(|e| e.to_string().to_ascii_lowercase())?;
    let size = decoder.dimensions();
    let color_space = ImageColorspace::Rgb;
    let icc = decoder
        .icc_profile()
        .map_err(|e| e.to_string().to_ascii_lowercase())?
        .and_then(|d| get_icc_profile_type(&d, color_space));

    Ok(ImageMetadata {
        has_alpha: decoder.has_alpha(),
        bits_per_component: BitsPerComponent::Eight,
        size,
        color_space,
        icc,
    })
}

fn decode_webp(data: Data) -> Result<Repr, String> {
    let mut decoder = image_webp::WebPDecoder::new(std::io::Cursor::new(data.as_ref()))
        .map_err(|e| e.to_string().to_ascii_lowercase())?;
    let mut first_frame = vec![0; decoder.output_buffer_size().ok_or("image is too large")?];
    decoder
        .read_image(&mut first_frame)
        .map_err(|e| e.to_string().to_ascii_lowercase())?;

    let color_space = if decoder.has_alpha() {
        ColorSpace::RGBA
    } else {
        ColorSpace::RGB
    };

    let (color_channel, alpha_channel, bits_per_component) =
        handle_u8_image(&first_frame, color_space);

    Ok(Repr::Sampled(SampledRepr {
        color_channel,
        alpha_channel,
        bits_per_component,
    }))
}

fn handle_u8_image(data: &[u8], cs: ColorSpace) -> (Vec<u8>, Option<Vec<u8>>, BitsPerComponent) {
    let mut alphas = if cs.has_alpha() {
        Vec::with_capacity(data.len() / cs.num_components())
    } else {
        Vec::new()
    };

    let color_channel = match cs {
        ColorSpace::RGB => deflate_encode(data),
        ColorSpace::RGBA => {
            let mut buf = Vec::with_capacity(data.len() * 3 / 4);
            data.chunks_exact(4).for_each(|data| {
                buf.extend_from_slice(&data[0..3]);
                alphas.push(data[3]);
            });
            deflate_encode(&buf)
        }
        ColorSpace::Luma => deflate_encode(data),
        ColorSpace::LumaA => {
            let mut buf = Vec::with_capacity(data.len() / 2);
            data.chunks_exact(2).for_each(|data| {
                buf.push(data[0]);
                alphas.push(data[1]);
            });
            deflate_encode(&buf)
        }
        // PNG/WEBP/GIF only support those three, so should be enough?
        _ => unimplemented!(),
    };

    let alpha_channel = if !alphas.is_empty() && alphas.iter().any(|v| *v != 255) {
        Some(deflate_encode(&alphas))
    } else {
        None
    };

    (color_channel, alpha_channel, BitsPerComponent::Eight)
}

fn handle_u16_image(data: &[u8], cs: ColorSpace) -> (Vec<u8>, Option<Vec<u8>>, BitsPerComponent) {
    let mut alphas = if cs.has_alpha() {
        Vec::with_capacity(data.len() / cs.num_components())
    } else {
        Vec::new()
    };

    let encoded_image = match cs {
        ColorSpace::RGB => deflate_encode(data),
        ColorSpace::RGBA => {
            let mut buf = Vec::with_capacity(data.len() * 3 / 4);
            data.chunks_exact(8).for_each(|data| {
                buf.extend_from_slice(&data[0..6]);
                alphas.extend_from_slice(&data[6..]);
            });
            deflate_encode(&buf)
        }
        ColorSpace::Luma => deflate_encode(data),
        ColorSpace::LumaA => {
            let mut buf = Vec::with_capacity(data.len() / 2);
            data.chunks_exact(4).for_each(|data| {
                buf.extend_from_slice(&data[0..2]);
                alphas.extend_from_slice(&data[2..]);
            });
            deflate_encode(&buf)
        }
        // PNG/WEBP/GIF only support those three, so should be enough?
        _ => unimplemented!(),
    };

    let encoded_mask = if !alphas.is_empty() {
        Some(deflate_encode(&alphas))
    } else {
        None
    };

    (encoded_image, encoded_mask, BitsPerComponent::Sixteen)
}

fn get_icc_profile_type(data: &[u8], color_space: ImageColorspace) -> Option<GenericICCProfile> {
    let wrapper = match color_space {
        ImageColorspace::Rgb => GenericICCProfile::Rgb(ICCProfile::new(data)?),
        ImageColorspace::Luma => GenericICCProfile::Luma(ICCProfile::new(data)?),
        ImageColorspace::Cmyk => GenericICCProfile::Cmyk(ICCProfile::new(data)?),
    };

    Some(wrapper)
}

#[cfg(test)]
mod tests {
    use crate::image::Image;
    use std::sync::Arc;

    #[test]
    fn invalid_png_image() {
        let e = Image::from_png(Arc::new(b"dfngiudfg".to_vec()).into(), false);
        assert!(e.is_err());
    }

    #[test]
    fn invalid_jpeg_image() {
        let e = Image::from_jpeg(Arc::new(b"dfngiudfg".to_vec()).into(), false);
        assert!(e.is_err());
    }
}
