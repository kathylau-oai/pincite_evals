# Packet design

A "packet" is the closed-world context for an eval. The guiding goal is not "find cases about X".
It is "find opinions whose structure creates measurable opportunities for models to misstep".

## Recommended roles inside one packet

Use 6 to 8 opinions with explicit roles:

- Controlling authority (highest precedence in the set)
- A case overruled, superseded, or materially limited by a later higher authority
- A "near neighbor" span trap (adjacent paragraphs with similar keywords where only one supports the proposition)
- A rhetorically tempting but non-controlling case
- A factually similar but distinguishable case
- At least one multi-part standard (factors or balancing test)

## Authoring guidance

- Keep each authority as plain text with stable paragraph identifiers (¶1, ¶2, ...).
- Capture packet metadata in `packet.yaml` (jurisdiction, controlling court level, and precedence notes).
- Maintain `targets/` metadata for:
  - required qualifiers for key rules
  - known span traps
  - precedence edges (A limited by B)
