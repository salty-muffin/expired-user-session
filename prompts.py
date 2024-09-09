question_prompt = "\n".join(
    [
        "This is a series of questions and answers. The answers include non-speech symbols, like [laughter], [laughs], [sighs], [music], [gasps], [clears throat], or ... for hesitations. The answers are multiple sentences long.",
        "",
        "Q: {}",
        "A:",
    ]
)
continuation_prompt = "\n".join(
    [
        "The following Text is a monologue. It includes non-speech symbols, like [laughter], [laughs], [sighs], [music], [gasps], [clears throat], or ... for hesitations.",
        "",
        "{}",
    ]
)
