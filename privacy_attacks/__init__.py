from privacy_attacks.base import AttackResult

__all__ = ["AttackResult", "MembershipAuditor"]


def __getattr__(name: str):
    if name == "MembershipAuditor":
        from privacy_attacks.auditor import MembershipAuditor

        return MembershipAuditor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
