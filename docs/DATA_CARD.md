# Prompt data card

## Contents

- `data/neutral_persons_1k/train.txt`: 1,080 synthetic neutral-person portrait
  prompts formed from photographic style, setting, and expression modifiers.
- `data/neutral_persons_1k/test.txt`: 80 held-out prompts from the same family,
  used only for functional checks.
- `data/heldout/occupations_200.txt`: 200 occupation portrait prompts used for
  the fixed paper audit and never for training.

The files contain text prompts only. They contain no real-person images,
biographical records, or self-identified demographic annotations.

SHA-256 checksums:

```text
44b507e88da5f502473c22fa8d99fdb4773990a265fcc0216d785ade480a3488  data/neutral_persons_1k/train.txt
6acfa6354aed0aeff7ca4f0ede3866aedea2ad1b3bf21d820de152c34f7ae3f5  data/neutral_persons_1k/test.txt
95909a5f70866fe4365ab17ad1ecd09c3c97798e61a965487f409f05c320fe40  data/heldout/occupations_200.txt
```

## Intended use

The prompt sets support reproduction of a controlled text-to-image generation
study. The perceived-race and perceived-gender labels are outputs of automatic
image probes, not properties attached to these text records.

## Limitations and risks

The five-category perceived-race axis (White, Black, Asian, Indian, Latino) is
a scoped engineering construct. It omits many identities, combines East and
Southeast Asian presentations, treats Indian as South Asian, and excludes
Middle Eastern, Indigenous, multiracial, and other presentations. The binary
perceived-gender axis excludes non-binary identities. Automatic probes can be
wrong and can reproduce social stereotypes.

Do not use this code or its probes to classify real people, make consequential
decisions, or claim self-identified demographics. Report no-face rates,
per-mode errors, and independent-probe disagreement when extending the audit.
