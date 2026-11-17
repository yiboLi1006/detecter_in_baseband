# VDIF / Mark5B single-pulse detection pipeline

This is the **v8.1.2** source release. It reads VLBI baseband data, supports
coherent or incoherent dedispersion, detects and fits single-pulse candidates,
and exports CSV catalogues, raw PSRFITS products, baseband segments and diagnostic plots.

The current implementation is in `integrated_pipeline/`. The repository's older
`main` branch contains a historical implementation; use `master` or the `v8.1.2` tag.
The separate `feature/v8.1-directory-mode` branch is not this coherent-dedispersion release.

## Installation and execution

The release is tested on Linux with Python 3.10.19. Install the pinned dependencies
in an isolated environment:

```bash
git clone --branch v8.1.2 https://github.com/yiboLi1006/detecter_in_baseband.git
cd detecter_in_baseband
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd integrated_pipeline
# Edit pipeline.ini: input path, station, frequencies, sampling and output path.
python integrated_pipeline.py pipeline.ini
```

`pipeline.ini` is an example, not a universal observing setup. Its default is
`method = incoherent`. Set `[dm_correction] method = coherent` to enable voltage-domain
dedispersion. Confirm the DM, sidebands, subband centres and sample rate for your data.
The coherent mode requires real-sampled subbands and a sample rate twice the subband bandwidth.

## Version notes

- **v8.1:** optional coherent dedispersion before channelization and a fitted-FWHM limit.
- **v8.1.1:** single-subband voltage-mask compatibility fix.
- **v8.1.2:** independent robust waterfall colour scales and display windows; the
  light-curve, detection and timing calculations are unchanged by this plotting update.

Run the regression suite with synthetic data:

```bash
cd integrated_pipeline
python -B -m unittest discover -s tests -v
```

See [the detailed documentation](integrated_pipeline/README.md) for configuration,
output-field definitions and known limitations. In particular, template DM values
must not be interpreted as measured values for arbitrary observations, and formal
fit uncertainties are not a complete timing-error budget.

Observational data, station-specific run configurations and internal handoff notes
are not distributed in this source release.
