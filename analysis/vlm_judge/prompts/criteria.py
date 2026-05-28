"""Per-criterion rubric templates used to prompt the VLM judge.

Each entry maps a TASTE sub-criterion slug (matching the slug used in
`distribution_tests/taste_stats.py:DIMENSIONS`) to:
  - display name the judge sees
  - one-paragraph rubric (the same text shown to designers, when known)
  - axis-specific question phrasing

The rubric strings here are short, operational versions of the
designer-facing instructions.  Once the data-collection team fills in
the §3.4 (Curation details TODO) hallucination/criterion rubric, copy
the exact text into this file so the VLM judge sees the same rubric
designers did.
"""

CRITERIA = {
    "aesthetics_preference": {
        "display": "UI+Ad Preference (holistic)",
        "rubric": (
            "Overall design quality for a UI mockup or ad creative.  "
            "Combine all aesthetic considerations into one judgement: "
            "which image looks like better professional design overall?"
        ),
        "question": "Which image is overall a better UI or ad design?",
        "question_variants": [
            "Which image is overall a better UI or ad design?",
            "Considering professional design quality, which image is stronger?",
            "Which image looks more like polished, professional graphic design?",
            "Holistically, which is the better design of the two?",
            "Which image would a graphic designer rate higher overall?",
            "Which image presents the more accomplished overall design?",
            "Which of these two designs is the stronger ad or UI piece?",
            "Taken as a whole, which image is the better piece of design?",
        ],
        "example_reasoning": (
            "Image A has muddy colours, awkward element spacing, and "
            "looks like a generic stock-photo collage.  Image B uses "
            "deliberate whitespace, a coherent type system, and a "
            "consistent palette that reads as professional design.  "
            "\n\n```json\n{\"answer\": \"B\"}\n```"
        ),
    },
    "aesthetics_mood": {
        "display": "Mood and Tone Match",
        "rubric": (
            "How well the image's mood, atmosphere, and emotional tone "
            "fit the brief.  Consider colour temperature, lighting, "
            "composition energy, and stylistic register.  Setting "
            "matters; an aggressive prompt should not look serene."
        ),
        "question": "Which image's mood and tone better match the brief?",
        "question_variants": [
            "Which image's mood and tone better match the brief?",
            "Which image's atmosphere is closer to what the brief asks for?",
            "Which image more accurately captures the emotional register the brief requests?",
            "Which image's stylistic mood fits the brief better?",
            "Whose mood and feel is more on-brief?",
            "Which image conveys the mood the brief describes more faithfully?",
            "Which image's emotional tone aligns better with the brief?",
            "Which image better realises the mood the brief specifies?",
        ],
        "example_reasoning": (
            "The brief asks for a vibrant, playful mood.  Image A is "
            "muted, with desaturated tones and a static composition "
            "that reads as corporate.  Image B uses bright primary "
            "colours, dynamic asymmetry, and a sense of motion that "
            "matches the requested playful register.  \n\n```json\n{\"answer\": \"B\"}\n```"
        ),
    },
    "aesthetics_visual_hier": {
        "display": "Visual Hierarchy",
        "rubric": (
            "Whether the image directs the eye through the composition "
            "in a clear order.  Strong visual hierarchy means primary "
            "elements are emphasised, secondary elements are grouped "
            "and subordinated, and the reading path is unambiguous."
        ),
        "question": "Which image has stronger visual hierarchy?",
        "question_variants": [
            "Which image has stronger visual hierarchy?",
            "Which image more clearly directs the viewer's eye through the composition?",
            "Which image has a more legible reading order?",
            "In which image are primary elements better emphasised over secondary ones?",
            "Which image's compositional hierarchy is clearer?",
            "Which image guides the eye through the design more deliberately?",
            "Which image has a more intentional visual emphasis structure?",
            "Which image presents a clearer hierarchy of design elements?",
        ],
        "example_reasoning": (
            "Image A presents the title, subtitle, body text, and CTA "
            "at similar weights and sizes; the eye has no clear entry "
            "point.  Image B has a dominant headline, well-grouped "
            "supporting text, and a contrasting CTA, so the reading "
            "path is unambiguous.  \n\n```json\n{\"answer\": \"B\"}\n```"
        ),
    },
    "aesthetics_color_harmony": {
        "display": "Colour Harmony",
        "rubric": (
            "How well the image's colours work together as a palette.  "
            "Strong colour harmony has intentional contrast, no jarring "
            "clashes, and a coherent overall palette."
        ),
        "question": "Which image has more harmonious colour use?",
        "question_variants": [
            "Which image has more harmonious colour use?",
            "Which image's palette is more coherent?",
            "Whose colour relationships are more intentionally designed?",
            "Which image has a more unified colour story?",
            "Which image avoids jarring colour clashes more effectively?",
            "Which image's colour palette works better as a whole?",
            "Which image uses colour more harmoniously?",
            "Which image's colour combinations feel more deliberate?",
        ],
        "example_reasoning": (
            "Image A pairs a saturated red against a muted teal with no "
            "transitional tones, producing a jarring contrast.  Image B "
            "uses a tighter palette of analogous oranges and yellows "
            "with a single deliberate complementary accent.  Final "
            "answer: B."
        ),
    },
    "aesthetics_typography": {
        "display": "Typography (Aesthetics)",
        "rubric": (
            "Typographic craft of any visible text in the image.  "
            "Consider font selection, spacing, sizing hierarchy, "
            "alignment, and whether the type style fits the design "
            "context."
        ),
        "question": "Which image has better typography?",
        "question_variants": [
            "Which image has better typography?",
            "Which image's typographic craft is stronger?",
            "Which image has more carefully crafted type?",
            "Which image's font selection and treatment is more skilful?",
            "Which image has a more accomplished typographic system?",
            "Whose text is set with more typographic care?",
            "Which image's type design is of higher quality?",
            "Which image's typography reads as more professional?",
        ],
        "example_reasoning": (
            "Image A mixes three different typefaces, with uneven "
            "kerning on the headline and inconsistent baselines.  "
            "Image B uses one consistent type family, deliberate "
            "weight contrast between headline and body, and tightly "
            "aligned baselines.  \n\n```json\n{\"answer\": \"B\"}\n```"
        ),
    },
    "descriptions_preference": {
        "display": "Description Preference (holistic)",
        "rubric": (
            "Overall faithfulness to the brief.  Which image more "
            "accurately and completely realises what the prompt asks "
            "for?  Combine all description-fidelity considerations."
        ),
        "question": "Which image better realises the brief overall?",
        "question_variants": [
            "Which image better realises the brief overall?",
            "Which image is more faithful to what the brief describes?",
            "Which image more completely captures the brief's specifications?",
            "Holistically, which image better matches the brief?",
            "Which image is the better realisation of the brief as a whole?",
            "Which image renders the brief's content more accurately overall?",
            "Considering the full brief, which image is the closer match?",
            "Which of these two images better executes what the brief asks for?",
        ],
        "example_reasoning": (
            "Image A omits one of the three required tiers and the "
            "background colour does not match the brief.  Image B "
            "includes all three tiers in the requested arrangement, "
            "with the correct background and the correct call-to-"
            "action button placement.  \n\n```json\n{\"answer\": \"B\"}\n```"
        ),
    },
    "descriptions_color_acc": {
        "display": "Colour Accuracy",
        "rubric": (
            "How accurately the image's colours match the colours "
            "specified in the prompt.  Look for the specific colours "
            "named or implied by the brief and judge whether they "
            "appear correctly."
        ),
        "question": "Which image more accurately uses the colours requested in the prompt?",
        "question_variants": [
            "Which image more accurately uses the colours requested in the prompt?",
            "Which image's colours match what the brief specifies more closely?",
            "Which image renders the requested palette more faithfully?",
            "Whose colour choices align better with the brief's specification?",
            "Which image gets the brief's named colours right?",
            "Which image is more accurate to the colours described?",
            "Which image more faithfully reproduces the colours called for in the brief?",
            "Which image's colour rendering is closer to the brief's requirements?",
        ],
        "example_reasoning": (
            "The brief specifies a 'warm orange-yellow background' "
            "and 'red building illustration'.  Image A uses a cool "
            "blue background and a green illustration.  Image B "
            "renders the warm orange-yellow background and red "
            "illustration as described.  \n\n```json\n{\"answer\": \"B\"}\n```"
        ),
    },
    "descriptions_spatial_acc": {
        "display": "Spatial Accuracy",
        "rubric": (
            "How accurately the image's spatial layout matches the "
            "spatial relationships described in the prompt.  Look for "
            "positional language (left, right, above, beside, in front "
            "of) and judge whether the layout matches."
        ),
        "question": "Which image more accurately reflects the spatial layout described in the prompt?",
        "question_variants": [
            "Which image more accurately reflects the spatial layout described in the prompt?",
            "Which image's element placement matches the brief's spatial directions more closely?",
            "Which image is more faithful to the positional language in the brief?",
            "Whose layout is closer to the spatial arrangement the brief describes?",
            "Which image more correctly arranges elements according to the brief?",
            "Which image's composition matches the spatial relationships in the brief?",
            "Which image places elements where the brief specifies?",
            "Which image gets the brief's left/right/above/below/beside relations right?",
        ],
        "example_reasoning": (
            "The brief asks for three cards arranged horizontally in "
            "a single row, with the centre card slightly taller.  "
            "Image A stacks the cards vertically in a column.  Image "
            "B arranges them in a horizontal row with the centre card "
            "elevated as specified.  \n\n```json\n{\"answer\": \"B\"}\n```"
        ),
    },
    "descriptions_typography": {
        "display": "Typography (Descriptions)",
        "rubric": (
            "Whether any text demanded by the prompt is rendered "
            "correctly.  Consider whether the requested words appear, "
            "whether they are spelled correctly, and whether typography "
            "choices fit the brief's directions."
        ),
        "question": "Which image more accurately renders the text demanded by the prompt?",
        "question_variants": [
            "Which image more accurately renders the text demanded by the prompt?",
            "Which image's text rendering is closer to what the brief specifies?",
            "Which image better matches the typography requirements in the brief?",
            "Which image renders the required text elements more faithfully?",
            "Whose text rendering is more accurate to the brief's specifications?",
            "Which image gets the brief's text elements right?",
            "Which image's typography aligns better with the prompt's text requirements?",
            "Which image is more faithful to the text content requested in the brief?",
        ],
        "example_reasoning": (
            "The brief requires the labels 'BASIC', 'COMFY', 'LUX' "
            "and the prices '$65', '$80', '$120'.  Image A renders "
            "'COMFY' as 'COMFFY' (extra F) and the LUX price is "
            "missing.  Image B spells all three labels correctly and "
            "shows all three prices as specified.  \n\n```json\n{\"answer\": \"B\"}\n```"
        ),
    },
}


PROMPT_STYLES = (
    "single_char", "reason_then_answer", "reason_with_synthetic_example",
)


def build_judge_prompt(criterion_slug: str, design_brief: str,
                       style: str = "reason_then_answer",
                       question_variant: int = 0) -> str:
    """Render a single judge prompt for the given criterion + brief.

    `style` controls how the model is asked to respond:
      - "single_char": legacy.  Replies "A" or "B" with no other text.
        Forces deterministic top-1 token; multi-sample protocols
        collapse to identical samples under this style.
      - "reason_then_answer": model writes 2-3 sentences of reasoning
        then a "Final answer: <X>" line.  Gives sampling room to vary,
        which is what we want for the 5-sample multi-rater protocol.
      - "reason_with_synthetic_example": as `reason_then_answer`, with
        a per-criterion text-only synthetic example (no images)
        demonstrating reasoning style.  All examples deliberately
        commit to "Final answer: B" so the model is inoculated
        against an "always A" position prior.
    """
    spec = CRITERIA[criterion_slug]
    variants = spec.get("question_variants") or [spec["question"]]
    chosen_q = variants[question_variant % len(variants)]
    base = (
        f"You are a professional graphic designer evaluating two AI-"
        f"generated images for the same brief.\n\n"
        f"BRIEF:\n{design_brief}\n\n"
        f"CRITERION: {spec['display']}\n"
        f"RUBRIC: {spec['rubric']}\n\n"
        f"You will see two images, labelled A and B.  Question: "
        f"{chosen_q}\n\n"
    )
    if style == "single_char":
        return base + (
            "Reply with exactly one character: 'A' if image A is better, "
            "'B' if image B is better.  Do not include any other text."
        )
    json_instr = (
        "End your response with a JSON code block containing the answer "
        "letter, in this exact format:\n\n"
        "```json\n"
        "{\"answer\": \"A\"}\n"
        "```\n\n"
        "(or `{\"answer\": \"B\"}`).  Do not write anything after the JSON "
        "code block."
    )
    if style == "reason_then_answer":
        return base + (
            "Briefly explain your reasoning in 2-3 sentences, grounded in "
            "the criterion and rubric above.  " + json_instr
        )
    if style == "reason_with_synthetic_example":
        example = spec.get("example_reasoning")
        if not example:
            raise ValueError(
                f"No example_reasoning defined for criterion "
                f"{criterion_slug!r}; cannot build "
                f"reason_with_synthetic_example prompt."
            )
        return base + (
            "Here is one example of the reasoning style and verdict "
            "format we expect.  This example does not refer to the "
            "actual images you will see; it is for format only:\n\n"
            "EXAMPLE REASONING (illustrative, not the real images):\n"
            f"{example}\n\n"
            "Now evaluate the actual two images for the brief above.  "
            "Briefly explain your reasoning in 2-3 sentences, grounded "
            "in the criterion and rubric above.  " + json_instr
        )
    raise ValueError(
        f"Unknown prompt style: {style!r}.  Expected one of {PROMPT_STYLES}."
    )
