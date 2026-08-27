# Benchmark Targeted Question Bank

Tools and pipelines for downloading, validating, and normalizing coding and reasoning benchmarks from Hugging Face into unified question banks.

## Structure

- **[`benchmark_downloader/`](./benchmark_downloader)**: Contains the benchmark downloader, normalizer, and comprehensive documentation.
  - [`download_and_normalize_benchmarks.py`](./benchmark_downloader/download_and_normalize_benchmarks.py): Download and normalize coding + reasoning benchmarks.
  - [`README.md`](./benchmark_downloader/README.md): Detailed usage instructions, CLI arguments, and schema specifications.
  - [`CAVEATS.md`](./benchmark_downloader/CAVEATS.md): Technical nuances, Hugging Face quirks, and edge-case handling.
