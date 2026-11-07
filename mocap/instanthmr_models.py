"""InstantHMR live-estimator weights, downloaded from Hugging Face at a pinned revision."""

REV = '07988f51c8c8908cfc7bb44fc18f645c9551b0bf'
_BASE = 'https://huggingface.co/momolesang/InstantHMR/resolve/' + REV + '/'
_PAGE = 'https://huggingface.co/momolesang/InstantHMR'


def _entry(key, name, filename, size, digest):
    return dict(key=key, name=name, relative='instanthmr/' + filename, url=_PAGE, minimum_bytes=size,
                download_url=_BASE + filename, expected_sha256=digest)


ASSETS = [
    _entry('instanthmr_medium', 'Live Body / InstantHMR medium (non-commercial)', 'instanthmr-medium.onnx',
           129345069, '937dbcc55a9ab6e8236dd5212ff02d57e94fed6e11fcc27d78ca70d81ea9efe5'),
    _entry('instanthmr_nano', 'Live Body / InstantHMR nano (non-commercial, smallest)', 'instanthmr-nano.onnx',
           80499078, 'a9e78fb89649d59bc8df294c6d23cd460168003b6419386bb053d353c3f85a69'),
    _entry('instanthmr_commercial', 'Live Body / InstantHMR medium-commercial (Apache-2.0)',
           'instanthmr-medium-commercial.onnx', 129345069,
           'f19d4492eba035337dc4fdc347cf3fe911570e73a765a70a35f9b32d7dec60dc'),
    _entry('instanthmr_body', 'Live Body / InstantHMR MHR body graph', 'mhr-body.onnx',
           12676712, 'eca2224d724f91b9e4f4bd7c86f6eafc74ab59ff0494bfb462e375cd04dac0fb'),
]
# Weight file per selectable variant; every variant also needs the body graph.
VARIANTS = {'medium': 'instanthmr_medium', 'nano': 'instanthmr_nano', 'commercial': 'instanthmr_commercial'}
DEFAULT_VARIANT = 'medium'
BODY = 'instanthmr_body'
