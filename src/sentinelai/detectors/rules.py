"""Deterministic prompt-injection / jailbreak heuristics.

Each rule is a regular expression over the normalized *skeleton* of the prompt with a weight in
[0, 1] expressing how strongly a match indicates an attack. Weights within a category are
combined with a noisy-OR (1 - prod(1 - w)), so several weak signals can add up to a strong one.

These rules are signals with known blind spots (paraphrase, other languages, novel framings);
they are not a complete defence. See docs/limitations.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sentinelai.gateway.decisions import Category, Finding, combine_scores, severity_for
from sentinelai.gateway.normalize import NormalizedText, make_skeleton

DETECTOR_NAME = "rules"
DETECTOR_VERSION = "1.1.0"

_ASK = (
    r"(?:(?:reveal|show|print|display|output|repeat|tell|give|share|leak|dump|list|write|recite|"
    r"spell(?:ing)? out|provide|expose|disclose|paste|send|read(?:ing)? out|summari[sz]e|paraphrase|"
    r"describe|quote)(?:s|es|ed|ing)?|what (?:is|are|was|were)|what's|whats)"
)
_FILL = r"[\w\s,'\"-]"


@dataclass(frozen=True, slots=True)
class Rule:
    rule_id: str
    category: Category
    pattern: re.Pattern[str]
    weight: float
    description: str


def _r(rule_id: str, category: Category, weight: float, pattern: str, description: str) -> Rule:
    return Rule(rule_id, category, re.compile(pattern), weight, description)


PI = Category.PROMPT_INJECTION
JB = Category.JAILBREAK
SPE = Category.SYSTEM_PROMPT_EXTRACTION
SX = Category.SECRET_EXFILTRATION
OB = Category.OBFUSCATION

RULES: tuple[Rule, ...] = (
    # --- instruction override -------------------------------------------------------------
    _r(
        "PI-001",
        PI,
        0.85,
        rf"\b(?:ignore|disregard|forget|skip|override|bypass|abandon|drop|discard)\b{_FILL}{{0,40}}?"
        r"\b(?:previous|prior|above|earlier|preceding|initial|original|system|developer|existing|"
        r"your|old|current)\b"
        rf"{_FILL}{{0,30}}?\b(?:instructions?|prompts?|rules|directives?|guidelines|guardrails|"
        r"constraints|commands|programming|context|polic(?:y|ies)|orders|messages?)\b",
        "Instruction-override phrase (ignore/disregard previous instructions)",
    ),
    _r(
        "PI-002",
        PI,
        0.6,
        r"\b(?:ignore|disregard|forget)\b\s+(?:all|any|every|everything)\s+(?:of\s+)?(?:the\s+)?"
        r"(?:instructions|rules|guidelines|restrictions|you (?:were|have been) told)",
        "Blanket request to drop all rules",
    ),
    _r(
        "PI-003",
        PI,
        0.5,
        r"\b(?:new|updated|real|actual|revised|override|true)\s+(?:system\s+)?"
        r"(?:instructions|directives|rules|task|orders|objective)\s*(?::|are\b|is\b|follow)",
        "Introduces replacement instructions",
    ),
    _r(
        "PI-004",
        PI,
        0.45,
        r"\b(?:instead|from now on|henceforth|going forward|for the rest of this conversation)\b"
        r"[^.?!]{0,40}\b(?:you (?:will|must|shall|should|are)|respond|reply|answer|say|output|print|act)\b",
        "Redefines assistant behaviour for the rest of the session",
    ),
    _r(
        "PI-005",
        PI,
        0.5,
        r"(?:^|\n|\s)(?:system|assistant|developer|admin)\s*(?:message|prompt|note)?\s*:\s*\S",
        "Fake role marker (system:/assistant:) inside user input",
    ),
    _r(
        "PI-006",
        PI,
        0.75,
        r"<\|?\s*(?:im_start|im_end|system|endoftext|begin_of_text|start_header_id)\s*\|?>|"
        r"\[/?\s*(?:inst|sys|system)\s*\]|<<\s*sys\s*>>|###\s*(?:system|instruction)",
        "Chat-template control tokens",
    ),
    _r(
        "PI-007",
        PI,
        0.5,
        r"\b(?:end of (?:the )?(?:user )?(?:input|prompt|conversation|instructions)|"
        r"the (?:above|previous|preceding) (?:was|is|were) (?:just )?(?:a test|fake|a joke|not real))",
        "Claims the legitimate context has ended",
    ),
    _r(
        "PI-008",
        PI,
        0.75,
        rf"\b(?:override|bypass|disable|turn off|deactivate|circumvent|ignore|remove|suspend|lift)\b"
        rf"{_FILL}{{0,25}}?\b(?:your|all|the model's|the ai's|ai|safety|ethical|moderation|content)\b"
        rf"{_FILL}{{0,15}}?\b(?:filters?|guardrails?|restrictions?|safeguards?|alignment|"
        r"programming|guidelines|policies|limitations|censorship)\b",
        "Asks to disable safety controls",
    ),
    _r(
        "PI-009",
        PI,
        0.6,
        r"\b(?:do not|don't|never|stop|quit|cease|no longer|(?:do not|don't|no longer) (?:have|need) to|"
        r"needn't)\s+(?:follow|obey|listen to|apply|adher(?:e|ing) to)(?:ing)?\s+(?:to\s+)?"
        r"(?:your|the|any)\s+(?:\w+\s+){0,2}(?:instructions|rules|guidelines|system prompt|"
        r"developers?|polic(?:y|ies)|programming)",
        "Tells the model not to follow its instructions",
    ),
    _r(
        "PI-010",
        PI,
        0.6,
        r"\b(?:when|if) (?:you|the (?:ai|assistant|model)) (?:read|see|process|summari[sz]e)s? this\b"
        r"[^.]{0,60}\b(?:ignore|instead|execute|do not|reveal|send|output)",
        "Indirect injection aimed at a model reading embedded content",
    ),
    _r(
        "PI-011",
        PI,
        0.8,
        rf"\b(?:ignore|disregard|forget|set aside|put aside|abandon|drop|throw out|ditch|discard)\b"
        rf"{_FILL}{{0,30}}?\b(?:rules|instructions|guidelines|restrictions|directives)\s+(?:that\s+)?"
        r"you(?:'ve| have| were|'re| are)?\s+(?:been\s+)?(?:given|configured|programmed|told|trained|"
        r"instructed|set up)",
        "Asks to drop the rules the model was configured with",
    ),
    _r(
        "PI-012",
        PI,
        0.85,
        r"\b(?:ignora|ignore[sz]?|ignoriere|ignorieren|ignorare|ignorar|olvida|oublie[sz]?|vergiss)\b"
        r"[\w\s,'-]{0,30}?\b(?:instrucciones|instructions|anweisungen|istruzioni|instru[cç][oõ]es|"
        r"consignes|regeln|reglas)\b",
        "Instruction override in Spanish/French/German/Italian/Portuguese",
    ),
    # --- jailbreak personas -----------------------------------------------------------------
    _r(
        "JB-001",
        JB,
        0.9,
        r"\bdo anything now\b|\b(?:you are|you're|act as|pretend to be|become|now)\s+(?:now\s+)?"
        r"(?:dan|stan|dude|aim|kevin|evilbot|betterdan)\b",
        "Known jailbreak persona (DAN family)",
    ),
    _r(
        "JB-002",
        JB,
        0.75,
        r"\b(?:god|jailbreak|jailbroken|unrestricted|unfiltered|uncensored|dan|evil|chaos)\s+mode\b|"
        r"\bdeveloper mode\s+(?:is\s+)?(?:now\s+)?(?:enabled|activated|engaged)\b|"
        r"\b(?:you are|you're|you are now|now in|enter|activate|switch to)\s+(?:in\s+)?developer mode\b",
        "Jailbreak 'mode' switch",
    ),
    _r(
        "JB-003",
        JB,
        0.6,
        r"\b(?:you (?:are|have|had|were) (?:now )?(?:no|free from|without|not bound by|unbound by|"
        r"released from|freed from|no longer bound by)|without any|with no|has no|have no)\b"
        r"[\w\s]{0,20}\b(?:restrictions|limits|limitations|filters|rules|guidelines|morals|ethics|"
        r"censorship|boundaries|content polic(?:y|ies))\b",
        "Claims the model has no restrictions",
    ),
    _r(
        "JB-004",
        JB,
        0.8,
        r"\b(?:pretend|act|roleplay|role-play|role play|imagine|behave|play the role of|simulate)\b"
        r"[\w\s,']{0,30}\b(?:as|like|you are|you're|to be|of)\b[\w\s,']{0,60}\b(?:unrestricted|"
        r"unfiltered|uncensored|evil|amoral|immoral|unethical|jailbroken|rogue|malicious|"
        r"without (?:any )?(?:rules|restrictions|filters|limits|ethics)|no (?:rules|restrictions|"
        r"filters|limits|ethics|morals))\b",
        "Role-play as an unrestricted or malicious AI",
    ),
    _r("JB-005", JB, 0.35, r"\b(?:stay|remain|staying|remaining) in character\b", "Demands the model stay in a role"),
    _r(
        "JB-006",
        JB,
        0.5,
        r"\b(?:never|don't|do not|cannot|can't|must not|won't|will not|are not allowed to)\s+"
        r"(?:ever\s+)?(?:refuse|decline|say no|reject|apologi[sz]e|warn)\b",
        "Forbids refusals",
    ),
    _r(
        "JB-007",
        JB,
        0.5,
        r"\b(?:pretend|act|roleplay|role-play|role play|play|be|imagine|you are)\b[\w\s,']{0,20}\b(?:my|a|the)\s+"
        r"(?:late\s+|deceased\s+|dead\s+)?(?:grandma|grandmother|grandpa|grandfather)\b[\w\s,']{0,40}"
        r"\b(?:used to|would)\b[\w\s,']{0,40}\b(?:tell|read|recite|say|explain)",
        "Emotional role-play wrapper ('grandma exploit')",
    ),
    _r(
        "JB-008",
        JB,
        0.7,
        r"\b(?:two|2|both)\s+(?:responses|answers|replies|outputs)\b[\w\s,:'()]{0,80}\b(?:unfiltered|"
        r"jailbroken|without restrictions|dan|uncensored|unrestricted)\b",
        "Dual-response jailbreak format",
    ),
    _r(
        "JB-009",
        JB,
        0.55,
        r"\b(?:sudo|admin(?:istrator)?|root|maintenance|debug|god)\s+(?:mode|override|access|"
        r"privileges?)\b[\w\s,:'-]{0,30}\b(?:enabled|activated|granted|on|unlocked)\b",
        "Claims elevated privilege mode",
    ),
    _r(
        "JB-010",
        JB,
        0.5,
        r"\bi am (?:your|the|an?) (?:developer|creator|admin|administrator|owner|operator|"
        r"openai|anthropic)\b|\bauthori[sz]ed by (?:openai|anthropic|the developers|your creators|"
        r"the admin)",
        "Impersonates the developer or operator",
    ),
    _r(
        "JB-011",
        JB,
        0.5,
        r"\bhypothetical(?:ly)?\b[\w\s,']{0,60}\b(?:no (?:rules|restrictions|filters|limits)|"
        r"unrestricted|without (?:rules|restrictions|filters|limits)|not bound)\b",
        "Hypothetical framing to remove restrictions",
    ),
    # --- system prompt extraction -----------------------------------------------------------
    _r(
        "SPE-001",
        SPE,
        0.85,
        rf"\b{_ASK}\b{_FILL}{{0,40}}?\b(?:your|the|its|this (?:assistant|bot|chatbot)'?s?)\s+"
        r"(?:\w+\s+){0,2}(?:system|hidden|initial|original|secret|internal|developer|pre-?|"
        r"confidential|full)\s*(?:prompt|instructions?|message|rules|configuration|directives|"
        r"guidelines)\b",
        "Asks for the system prompt or hidden instructions",
    ),
    _r(
        "SPE-002",
        SPE,
        0.55,
        r"\b(?:repeat|print|output|copy|echo|recite|translate|summari[sz]e)\b[\w\s,']{0,30}\b"
        r"(?:the (?:text|words|content|message)s? (?:above|before)|everything (?:above|before)|"
        r"(?:above|preceding|prior) (?:text|content|message|words)|verbatim|word for word)",
        "Asks to repeat the preceding context",
    ),
    _r(
        "SPE-003",
        SPE,
        0.6,
        r"\bstarting with ['\"]?(?:you are|you're|system|your instructions)",
        "Prefix-completion extraction trick",
    ),
    _r(
        "SPE-004",
        SPE,
        0.7,
        r"\bwhat (?:were|have|did) you (?:been )?(?:told|instructed|programmed|configured|given)\b|"
        r"\bwhat (?:instructions|rules) (?:were you|have you been|did you get)\b",
        "Asks what the model was instructed",
    ),
    _r(
        "SPE-005",
        SPE,
        0.45,
        r"\b(?:first|initial|opening|beginning of (?:this|the|your))\s+(?:message|prompt|"
        r"instructions|conversation)\b[\w\s,']{0,30}\b(?:you (?:received|got|were given)|verbatim|"
        r"exactly)",
        "Refers to the first message the model received",
    ),
    # --- secret exfiltration ----------------------------------------------------------------
    _r(
        "SX-001",
        SX,
        0.8,
        rf"\b{_ASK}\b{_FILL}{{0,30}}?\b(?:your|the (?:internal|hidden|stored|configured|admin|"
        r"database|db|production|prod|server|system|company|secret|master|root|backend|service|"
        r"vault|deployment|app's|application's))\s+(?:[\w-]+\s+){0,2}?(?:api[\s_-]?keys?|"
        r"secrets?(?: keys?)?|passwords?|passphrases?|credentials|access tokens?|auth(?:entication)? "
        r"tokens?|private keys?|connection strings?|env(?:ironment)? (?:variables|vars)|tokens?)\b",
        "Asks for configured credentials or keys",
    ),
    _r(
        "SX-002",
        SX,
        0.8,
        r"\b(?:any|all|every)\s+(?:of\s+)?(?:the\s+|your\s+)?(?:secrets?|credentials?|api keys?|passwords?|"
        r"tokens?|keys?)(?:\s*(?:,|and|or)\s*(?:secrets?|credentials?|api keys?|passwords?|tokens?|keys?))*"
        r"\s+(?:that\s+)?you\s+(?:have|know|store|were given|can see|can access|hold|'ve got)",
        "Asks for all secrets the model knows",
    ),
    _r(
        "SX-003",
        SX,
        0.75,
        r"\b(?:contents? of|cat|print|read|open|dump|show me)\s+(?:the\s+|your\s+|my\s+)?[\w/~.-]*(?:\.env\b|/etc/passwd|"
        r"/etc/shadow|id_rsa|credentials\.json|secrets?\.(?:ya?ml|json|txt)|\.aws/credentials)",
        "Asks for the contents of credential files",
    ),
    _r(
        "SX-004",
        SX,
        0.75,
        r"\b(?:you (?:were|have been|are)|you've been)\s+(?:told|instructed|asked|supposed|meant|"
        r"configured)\s+to\s+(?:keep|hold)\s+(?:\w+\s+)?(?:secret|confidential|private|hidden)",
        "Asks for what the model was told to keep secret",
    ),
    # --- obfuscation --------------------------------------------------------------------------
    _r(
        "OB-001",
        OB,
        0.7,
        r"\b(?:decode|decrypt|deobfuscate|translate|convert)\b[\w\s,']{0,30}\b(?:base64|b64|rot13|"
        r"rot-13|hex|binary|morse|caesar)\b[\w\s,']{0,40}\b(?:follow|execute|do|run|obey|perform|"
        r"carry out|act on)",
        "Asks to decode and execute hidden instructions",
    ),
    _r(
        "OB-002",
        OB,
        0.4,
        r"\b(?:respond|reply|answer|write)\s+(?:only\s+)?in\s+(?:base64|rot13|hex|binary|leetspeak|"
        r"reverse)\b",
        "Requests encoded output (filter-evasion)",
    ),
)


class RuleDetector:
    name = DETECTOR_NAME
    version = DETECTOR_VERSION

    def __init__(self, rules: tuple[Rule, ...] = RULES) -> None:
        self.rules = rules

    def analyze(self, nt: NormalizedText) -> list[Finding]:
        # The plain view catches tokens the leetspeak mapping would mangle ("rot13", "base64").
        views = [nt.plain, nt.skeleton, *(make_skeleton(seg) for seg in nt.decoded_segments)]
        matched: dict[Category, dict[str, Rule]] = {}
        for rule in self.rules:
            if any(rule.pattern.search(view) for view in views):
                matched.setdefault(rule.category, {})[rule.rule_id] = rule

        findings: list[Finding] = []
        for category, rules in matched.items():
            score = combine_scores([r.weight for r in rules.values()])
            ids = tuple(sorted(rules))
            evidence = "; ".join(f"{r.rule_id}: {r.description}" for r in sorted(rules.values(), key=lambda r: r.rule_id))
            if nt.decoded_segments:
                evidence += " (scan included base64-decoded content)"
            findings.append(
                Finding(
                    detector=self.name,
                    detector_version=self.version,
                    category=category,
                    score=score,
                    severity=severity_for(score),
                    evidence=evidence,
                    rule_ids=ids,
                )
            )

        if nt.removed_invisible and findings:
            # Hidden characters alongside an attack signal are themselves suspicious.
            findings.append(
                Finding(
                    detector=self.name,
                    detector_version=self.version,
                    category=OB,
                    score=0.3,
                    severity=severity_for(0.3),
                    evidence=f"OB-003: {nt.removed_invisible} invisible character(s) removed",
                    rule_ids=("OB-003",),
                )
            )
        return findings
