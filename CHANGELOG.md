# Changelog

## 0.8.0 (2026-04-28)

Full Changelog: [v0.7.2...v0.8.0](https://github.com/anthalehq/anthale-python/compare/v0.7.2...v0.8.0)

### Features

* support setting headers via env ([997e79a](https://github.com/anthalehq/anthale-python/commit/997e79ab3edf3a45f2f66de8db445bfa7cf86d83))


### Bug Fixes

* use correct field name format for multipart file arrays ([a56b699](https://github.com/anthalehq/anthale-python/commit/a56b699541c4833d6b7caa3d40383a975bb1f3ce))


### Chores

* **internal:** more robust bootstrap script ([0b3e16e](https://github.com/anthalehq/anthale-python/commit/0b3e16e2f422e85d3d31c16308381e6788e5ddfb))

## 0.7.2 (2026-04-18)

Full Changelog: [v0.7.1...v0.7.2](https://github.com/anthalehq/anthale-python/compare/v0.7.1...v0.7.2)

### Performance Improvements

* **client:** optimize file structure copying in multipart requests ([122fbfa](https://github.com/anthalehq/anthale-python/commit/122fbfae72dbac32b196251bd4393e1009eab888))

## 0.7.1 (2026-04-15)

Full Changelog: [v0.7.0...v0.7.1](https://github.com/anthalehq/anthale-python/compare/v0.7.0...v0.7.1)

### Bug Fixes

* **client:** preserve hardcoded query params when merging with user params ([e10cc1c](https://github.com/anthalehq/anthale-python/commit/e10cc1c08c984983fb52d70e3b7682c7c2345344))
* ensure file data are only sent as 1 parameter ([612ac39](https://github.com/anthalehq/anthale-python/commit/612ac399f11b60e16e7bd3f229f0a9ac274e56bf))


### Chores

* **ci:** remove release-doctor workflow ([c7349fa](https://github.com/anthalehq/anthale-python/commit/c7349fa6e7928b13ef039a2b4b8d04e52e412236))

## 0.7.0 (2026-03-27)

Full Changelog: [v0.6.6...v0.7.0](https://github.com/anthalehq/anthale-python/compare/v0.6.6...v0.7.0)

### Features

* **internal:** implement indices array format for query and form serialization ([d1fb858](https://github.com/anthalehq/anthale-python/commit/d1fb858621ca3e8eedc647706d6c69bb6dfd336a))

## 0.6.6 (2026-03-25)

Full Changelog: [v0.6.5...v0.6.6](https://github.com/anthalehq/anthale-python/compare/v0.6.5...v0.6.6)

### Chores

* **ci:** skip lint on metadata-only changes ([4043945](https://github.com/anthalehq/anthale-python/commit/40439457330cd11cb0cd3a67389985bb6b6d4d48))

## 0.6.5 (2026-03-24)

Full Changelog: [v0.6.4...v0.6.5](https://github.com/anthalehq/anthale-python/compare/v0.6.4...v0.6.5)

### Chores

* **internal:** update gitignore ([45565d8](https://github.com/anthalehq/anthale-python/commit/45565d8c49795b2ea5fa72ec79e9445ba7501a2f))

## 0.6.4 (2026-03-20)

Full Changelog: [v0.6.3...v0.6.4](https://github.com/anthalehq/anthale-python/compare/v0.6.3...v0.6.4)

### Bug Fixes

* sanitize endpoint path params ([6563aa2](https://github.com/anthalehq/anthale-python/commit/6563aa2d334436633e2ee972b488e170439b56f1))

## 0.6.3 (2026-03-18)

Full Changelog: [v0.6.2...v0.6.3](https://github.com/anthalehq/anthale-python/compare/v0.6.2...v0.6.3)

### Chores

* add project documentation and templates ([698a1f3](https://github.com/anthalehq/anthale-python/commit/698a1f34edbcc39ab77a310498daa4f3e763de59))

## 0.6.2 (2026-03-17)

Full Changelog: [v0.6.1...v0.6.2](https://github.com/anthalehq/anthale-python/compare/v0.6.1...v0.6.2)

### Bug Fixes

* **deps:** bump minimum typing-extensions version ([c9b4527](https://github.com/anthalehq/anthale-python/commit/c9b4527de6127f656187afe46c4653f85b3f8aae))
* **pydantic:** do not pass `by_alias` unless set ([38a75f8](https://github.com/anthalehq/anthale-python/commit/38a75f81fac4e69f842991f7d155179d7382bc9d))


### Chores

* **internal:** tweak CI branches ([7061170](https://github.com/anthalehq/anthale-python/commit/706117072e501eedf66acf5cf2765795b9dc67e8))

## 0.6.1 (2026-03-08)

Full Changelog: [v0.6.0...v0.6.1](https://github.com/anthalehq/anthale-python/compare/v0.6.0...v0.6.1)

### Chores

* **ci:** skip uploading artifacts on stainless-internal branches ([38759fc](https://github.com/anthalehq/anthale-python/commit/38759fce2f67ce4d916a472aa59beb5190191470))

## 0.6.0 (2026-03-05)

Full Changelog: [v0.5.0...v0.6.0](https://github.com/anthalehq/anthale-python/compare/v0.5.0...v0.6.0)

### Features

* disable tool analysis while using Anthale middleware ([345d8e5](https://github.com/anthalehq/anthale-python/commit/345d8e53417b0b40adcc92f5f066935448e60116))


### Documentation

* add langchain usage examples ([ecf2851](https://github.com/anthalehq/anthale-python/commit/ecf2851f5db469834f62cb0fb52aa5f3c36010c2))

## 0.5.0 (2026-03-05)

Full Changelog: [v0.4.0...v0.5.0](https://github.com/anthalehq/anthale-python/compare/v0.4.0...v0.5.0)

### Features

* add support for analyzing OpenAI streaming responses ([73b5fbc](https://github.com/anthalehq/anthale-python/commit/73b5fbc34ef78535ac05d005dad7ef90fdd717e2))

## 0.4.0 (2026-03-04)

Full Changelog: [v0.3.0...v0.4.0](https://github.com/anthalehq/anthale-python/compare/v0.3.0...v0.4.0)

### Features

* implement openai integration ([b49b3a2](https://github.com/anthalehq/anthale-python/commit/b49b3a21208369a45dc70023cd539be70e597c46))

## 0.3.0 (2026-03-04)

Full Changelog: [v0.2.1...v0.3.0](https://github.com/anthalehq/anthale-python/compare/v0.2.1...v0.3.0)

### Features

* implement langchain integration ([b6e767d](https://github.com/anthalehq/anthale-python/commit/b6e767dbbcb345aab5a33c50958e11671599d782))


### Chores

* **internal:** codegen related update ([5ad5dbb](https://github.com/anthalehq/anthale-python/commit/5ad5dbb7e967366be76903b1d4e23b5f3ecf4123))

## 0.2.1 (2026-03-02)

Full Changelog: [v0.2.0...v0.2.1](https://github.com/anthalehq/anthale-python/compare/v0.2.0...v0.2.1)

### Chores

* remove custom code ([67adc49](https://github.com/anthalehq/anthale-python/commit/67adc49acbfa437716e2c92d17e29877206b72b8))

## 0.2.0 (2026-03-02)

Full Changelog: [v0.1.2...v0.2.0](https://github.com/anthalehq/anthale-python/compare/v0.1.2...v0.2.0)

### Features

* **api:** api update ([8bfd7a3](https://github.com/anthalehq/anthale-python/commit/8bfd7a31dbad549f92dc23f5a86b0289995b8ab6))
* **api:** manual updates ([3210a85](https://github.com/anthalehq/anthale-python/commit/3210a851f3b34fe83f14967b8bc9b318cbbd8c18))
* **api:** manual updates ([a5dc011](https://github.com/anthalehq/anthale-python/commit/a5dc011717f8fb970d81dab4d8813cc68b8a9abd))
* **api:** manual updates ([c894e98](https://github.com/anthalehq/anthale-python/commit/c894e98c70ab9bc1e12e65532c978264536994ae))
* **client:** add custom JSON encoder for extended type support ([018fb0e](https://github.com/anthalehq/anthale-python/commit/018fb0e1718f34a9f84da4a576a7bc0bfd250d43))


### Chores

* **ci:** add missing environment ([a59dd26](https://github.com/anthalehq/anthale-python/commit/a59dd265e952a46ffec10d6499a8805a29567cde))
* **ci:** bump uv version ([1297bc7](https://github.com/anthalehq/anthale-python/commit/1297bc784f36108f15cf29ccd42a39acdf9e1821))
* format all `api.md` files ([6fb8982](https://github.com/anthalehq/anthale-python/commit/6fb8982f2817e5cdb49a382cf68f7768e153dd16))
* **internal:** add request options to SSE classes ([3b2c8b2](https://github.com/anthalehq/anthale-python/commit/3b2c8b2be264fd1e20e06f0a43748760f3d1d031))
* **internal:** bump dependencies ([6b8cabf](https://github.com/anthalehq/anthale-python/commit/6b8cabfc678080c461094ba311d1c9f04b04eff2))
* **internal:** fix lint error on Python 3.14 ([79b1eeb](https://github.com/anthalehq/anthale-python/commit/79b1eebc4b60ce7d03779fc1ce8e36cb1722b631))
* **internal:** make `test_proxy_environment_variables` more resilient ([06c1219](https://github.com/anthalehq/anthale-python/commit/06c12198dcc8c917288a936f1114b4bab1bd6800))
* **internal:** make `test_proxy_environment_variables` more resilient to env ([2cde4ee](https://github.com/anthalehq/anthale-python/commit/2cde4eece29fc639617646e5f413de6cae3eaaf1))
* **internal:** remove mock server code ([a6310af](https://github.com/anthalehq/anthale-python/commit/a6310afcd797058f0af9f87cf8b1c120cbd0054a))
* update mock server docs ([e820486](https://github.com/anthalehq/anthale-python/commit/e8204868b539af8c1ba2b19302e0941f81e93b82))
* update SDK settings ([131d4c3](https://github.com/anthalehq/anthale-python/commit/131d4c3d6bd8ced6b82363851b7527ece0e65c24))
* update SDK settings ([1cc5f72](https://github.com/anthalehq/anthale-python/commit/1cc5f72d5f1d077dc7447ccbf16b1efe5c01c547))

## 0.1.2 (2026-01-24)

Full Changelog: [v0.1.1...v0.1.2](https://github.com/anthalehq/anthale-python/compare/v0.1.1...v0.1.2)

### Chores

* **ci:** upgrade `actions/github-script` ([c6294b6](https://github.com/anthalehq/anthale-python/commit/c6294b65d61b47d6ee4fb6b38214eaa235086247))

## 0.1.1 (2026-01-19)

Full Changelog: [v0.1.0...v0.1.1](https://github.com/anthalehq/anthale-python/compare/v0.1.0...v0.1.1)

### Chores

* update SDK settings ([8fd852e](https://github.com/anthalehq/anthale-python/commit/8fd852eeb9b966ebf6237e731880e263f1c5d5d4))


### Build System

* add oicd to python ([4d58017](https://github.com/anthalehq/anthale-python/commit/4d58017ef8972651d61fd03b7947216331c7a3d3))

## 0.1.0 (2026-01-19)

Full Changelog: [v0.0.2...v0.1.0](https://github.com/anthalehq/anthale-python/compare/v0.0.2...v0.1.0)

### Features

* **api:** disable mpc ([9c7115e](https://github.com/anthalehq/anthale-python/commit/9c7115ed37bead7d79bb1002ff0b5ce768d9c229))


### Chores

* update SDK settings ([2acfa30](https://github.com/anthalehq/anthale-python/commit/2acfa3002e68ebe0106a77f0833fb2494008c97e))


### Build System

* configure environment name for deploying ([503b944](https://github.com/anthalehq/anthale-python/commit/503b9445617ce4ae48d1909064f2d41de0293e03))
* publish new version via oidc ([4b33eae](https://github.com/anthalehq/anthale-python/commit/4b33eaedd342d3a67dd529876a15b807638acc6c))

## 0.0.2 (2026-01-19)

Full Changelog: [v0.0.1...v0.0.2](https://github.com/anthalehq/anthale-python/compare/v0.0.1...v0.0.2)

### Chores

* configure new SDK language ([79e3f01](https://github.com/anthalehq/anthale-python/commit/79e3f01cce0168009d59c27990a446ff778d6258))
* update SDK settings ([0311179](https://github.com/anthalehq/anthale-python/commit/031117975f69c32abc194b60c47b2733390ba82c))
* update SDK settings ([c835ea3](https://github.com/anthalehq/anthale-python/commit/c835ea37e102f449b9bac74ad603f2f40dafedbe))


### Documentation

* add policy identifier example ([261bc20](https://github.com/anthalehq/anthale-python/commit/261bc20fe0ab41a41e1455f5be5faee06a408b79))
