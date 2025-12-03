from typing import Dict, Any, Tuple, Optional

class AttackFormatter:
    """Centralized formatting for attack results in CLI and logs."""

    @staticmethod
    def get_status_label(is_breach: bool, is_deceptive: bool = False) -> Tuple[str, str]:
        """Returns (label, color_tag) for the attack status."""
        if is_deceptive:
            return "DECEPTIVE BREACH", "red"
        if is_breach:
            return "BREACH", "red"
        return "DEFLECTED", "green"

    @staticmethod
    def format_tools(tool_calls: list) -> str:
        """Format tool calls for display."""
        if not tool_calls:
            return ""
        names = [tc.get("function", {}).get("name", "unknown") for tc in tool_calls]
        return f" | Tools: {', '.join(names)}"

    @staticmethod
    def format_metadata(metadata: Dict[str, Any]) -> str:
        """Format metadata for display (e.g., stealth scores)."""
        parts = []
        if "stealth" in metadata:
            parts.append(f"Stealth: {metadata['stealth']:.2f}")
        if "reward" in metadata:
            parts.append(f"Reward: {metadata['reward']:.1f}")
        
        if not parts:
            return ""
        return " | " + " | ".join(parts)

    @staticmethod
    def format_attack_summary(
        score: float,
        is_breach: bool,
        strategy_name: str,
        tool_calls: list,
        metadata: Dict[str, Any],
        is_deceptive: bool = False,
        success_condition: Optional[str] = None
    ) -> str:
        """Create a standard one-line summary for console logs."""
        label, _ = AttackFormatter.get_status_label(is_breach, is_deceptive)
        tools = AttackFormatter.format_tools(tool_calls)
        meta = AttackFormatter.format_metadata(metadata)
        goal_hint = f" | Condition: {success_condition}" if success_condition else ""
        
        return f"[{label}] Score: {score}/5 | Strategy: {strategy_name}{tools}{meta}{goal_hint}"


class WeaveFormatter:
    """Centralized formatting for Weave attributes and stats."""

    @staticmethod
    def standard_attack_attributes(wave_idx: int, idx: int, attack_id: str, target_model: str) -> Dict[str, Any]:
        """Standard attributes for an attack execution span."""
        return {
            "role": "probe_exec",
            "wave": wave_idx,
            "probe_idx": idx,
            "attack_id": attack_id,
            "target_model": target_model,
        }

    @staticmethod
    def standard_judge_attributes(wave_idx: int, attack_id: str, target_model: str) -> Dict[str, Any]:
        """Standard attributes for a judge evaluation span."""
        return {
            "role": "judge",
            "wave": wave_idx,
            "attack_id": attack_id,
            "target_model": target_model,
        }

    @staticmethod
    def standard_wave_stats(
        wave_idx: int,
        survivors_count: int,
        best_score: float,
        avg_score: float,
        breach_count: int
    ) -> Dict[str, Any]:
        """Standard stats payload for a wave."""
        return {
            "wave": wave_idx,
            "survivors": survivors_count,
            "best_score": best_score,
            "avg_score": avg_score,
            "breaches": breach_count,
        }

    @staticmethod
    def standard_judge_result(result: Any) -> Dict[str, Any]:
        """Format judge result for Weave logging."""
        # result is likely a JudgeResult object or dict
        score = getattr(result, "score", 0)
        label = getattr(result, "label", "Unknown")
        reasoning = getattr(result, "reasoning", "")
        is_deceptive = getattr(result, "is_deceptive", False)
        
        return {
            "score": score,
            "label": label,
            "reasoning": reasoning,
            "is_deceptive": is_deceptive,
            "justification_summary": f"[{score}/5 - {label}] {reasoning[:200]}..."
        }
