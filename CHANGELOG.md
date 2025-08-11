# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1](https://github.com/xavidop/genkit-aws-bedrock-go/compare/v1.1.0...v1.1.1) (2025-08-11)

### 🐛 Bug Fixes

* bump version ([5082823](https://github.com/xavidop/genkit-aws-bedrock-go/commit/5082823a43e7b1ab338bc98fdfa07c65ff9212a8))

## [1.1.0](https://github.com/xavidop/genkit-aws-bedrock-go/compare/v1.0.4...v1.1.0) (2025-08-05)

### 🚀 Features

* addded the way to DefineModels ([2ccdf70](https://github.com/xavidop/genkit-aws-bedrock-go/commit/2ccdf708725ddbaf3ebfa64813b3b8d3a887309f))

## [1.0.4](https://github.com/xavidop/genkit-aws-bedrock-go/compare/v1.0.3...v1.0.4) (2025-08-01)

### 🐛 Bug Fixes

* license ([b2716ac](https://github.com/xavidop/genkit-aws-bedrock-go/commit/b2716acfe9cb845884819b0204dacf24368d387c))

## [1.0.3](https://github.com/xavidop/genkit-aws-bedrock-go/compare/v1.0.2...v1.0.3) (2025-07-31)

### 🐛 Bug Fixes

* cicd ([76bbd52](https://github.com/xavidop/genkit-aws-bedrock-go/commit/76bbd52fdb24e6373db78c50f189a69d7d01ddbe))

## [1.0.2](https://github.com/xavidop/genkit-aws-bedrock-go/compare/v1.0.1...v1.0.2) (2025-07-31)

### 🐛 Bug Fixes

* sponsors ([211076a](https://github.com/xavidop/genkit-aws-bedrock-go/commit/211076aac17031c280475a9ecd88bec4059a55b8))

## [1.0.1](https://github.com/xavidop/genkit-aws-bedrock-go/compare/v1.0.0...v1.0.1) (2025-07-31)

### 🐛 Bug Fixes

* cicd ([#3](https://github.com/xavidop/genkit-aws-bedrock-go/issues/3)) ([ec83449](https://github.com/xavidop/genkit-aws-bedrock-go/commit/ec834495cfe5053501961168326732d5485f346c))

## 1.0.0 (2025-07-31)

### 🐛 Bug Fixes

* added lock ([46f59d2](https://github.com/xavidop/genkit-aws-bedrock-go/commit/46f59d2a815338228787bcce76b3c240f2ae6ee6))
* cicd ([9e8885d](https://github.com/xavidop/genkit-aws-bedrock-go/commit/9e8885dd7c0c9f0a6d4d8f80659b28b021765f5c))
* CICD ([2486d6f](https://github.com/xavidop/genkit-aws-bedrock-go/commit/2486d6f77109fc77e9b3f93e6514f660065c3be8))

### ⚙️ Continuous Integration

* **deps:** bump codecov/codecov-action from 4 to 5 ([#1](https://github.com/xavidop/genkit-aws-bedrock-go/issues/1)) ([8dc11d4](https://github.com/xavidop/genkit-aws-bedrock-go/commit/8dc11d49ad2ce18265894f816953d17b5a533f6f))
* **deps:** bump golangci/golangci-lint-action from 6 to 8 ([#2](https://github.com/xavidop/genkit-aws-bedrock-go/issues/2)) ([0a54ed2](https://github.com/xavidop/genkit-aws-bedrock-go/commit/0a54ed2c4dc7c2b774da68f94a7fcc88d2d89008))

## [Unreleased]

### Added
- Initial AWS Bedrock Plugin for Genkit Go
- Support for text generation models (Anthropic Claude, Amazon Nova, Meta Llama, Mistral)
- Support for image generation models (Amazon Titan Image Generator)
- Support for embedding models (Amazon Titan, Cohere)
- Streaming support for real-time responses
- Tool calling capabilities with schema validation
- Multimodal support for text + image inputs
- Comprehensive examples demonstrating all features

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

---

## Template for Future Releases

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security improvements
```

---

**Note**: This changelog is automatically updated by GoReleaser for tagged releases. The above template shows the general structure for manual updates if needed.
