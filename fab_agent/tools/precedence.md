<!--
Precedence framing template for get_precedence_context().
Edit this file to adjust the "defeasible reasoning" context handed to the agent.
Placeholders {setting} and {rel_level} are substituted at render time.
The CASUAL / TOURNAMENT blocks are selected by the `setting` argument; only the
matching block is included in the rendered output.
-->
# Applicable precedence

Flesh and Blood rulings are **defeasible**: a general rule holds *unless* a more
specific authority overrides it. Resolve conflicts by applying these authorities
in order of precedence.

1. **Comprehensive Rules (CR) govern by default.** Every game of Flesh and Blood
   follows the CR unless something below overrides it.
2. **Card text overrides general CR rules where they directly conflict.** A
   specific instruction printed on a card defeats the general rule it
   contradicts; the effect supersedes the rule (CR 1.0.1a).
3. **Restriction over requirement over allowance.** When effects conflict, a
   restriction ("cannot") beats a requirement ("must"), which beats an
   allowance ("may") (CR 1.0.2).

Setting: **{setting}**. Active Rules Enforcement Level (REL): **{rel_level}**.

<!-- CASUAL -->
This is **casual, non-sanctioned play**. No tournament policy applies: the
Comprehensive Rules and card text alone determine the ruling. Rules enforcement
is lenient and education-focused, so favor teaching the correct play over
punitive procedure. Cite the CR (and card text) as the governing authority.
<!-- /CASUAL -->

<!-- TOURNAMENT -->
This is **sanctioned tournament play**. Layered on top of the CR, two tournament
documents prevail **within their own scope**:

- **Tournament Rules and Policy (TRP)** governs tournament structure, player
  responsibilities, and legal game actions in an event.
- **Procedures and Penalty Guide (PPG)** governs infractions, the procedures to
  remedy them, and the penalties to apply.

For a *game-rules* question, the CR (as overridden by card text) still decides
what is legal; the TRP/PPG then determine the tournament consequence. Apply the
procedures and penalties appropriate to the **{rel_level}** REL, which sets how
strictly infractions are enforced.
<!-- /TOURNAMENT -->
