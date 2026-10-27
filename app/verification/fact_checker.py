import math

from app.knowledge_graph.graph import GRAPH_LOCK


class FactChecker:
    """
    Validate knowledge-graph structure and metadata.

    This component checks structural integrity. It does not determine
    whether evidence scientifically proves or entails a finding.

    The legacy 'verified' field is conservatively kept False because
    scientific support is not assessed by this component.
    Use 'structural_verified' to check structural validity.
    """

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def _response(
        self,
        finding_id,
        checks,
        *,
        reason,
        confidence=0.0,
        evidence=None,
        models=None,
    ):
        """Build a consistent verification response."""

        structural_verified = all([
            checks["finding_exists"],
            checks["is_finding"],
            checks["has_confidence"],
            checks["has_evidence"],
            checks["evidence_connected"],
            checks["valid_evidence_content"],
            checks["valid_confidence"],
            checks["valid_model_accuracy"],
        ])

        return {
            "finding_id": finding_id,

            # Scientific support has not been assessed.
            # Structural validity alone must not imply scientific truth.
            "verified": False,
            "structural_verified": structural_verified,
            "verification_scope": "structural_only",
            "scientific_support_assessed": False,

            "reason": reason,
            "confidence": confidence,
            "evidence": evidence or [],
            "models": models or [],
            "checks": checks,
        }

    def verify_finding(self, finding_id):
        """Check the structure and metadata of a finding."""

        with GRAPH_LOCK:
            finding_exists = finding_id in self.kg.graph.nodes

            # Initialize a consistent set of checks for early failures.
            checks = {
                "finding_exists": finding_exists,
                "is_finding": False,
                "has_confidence": False,
                "has_evidence": False,
                "evidence_connected": False,
                "valid_evidence_content": False,
                "has_model": False,
                "valid_confidence": False,
                "valid_model_accuracy": False,
            }

            if not finding_exists:
                return self._response(
                    finding_id,
                    checks,
                    reason="Finding does not exist.",
                )

            finding = dict(self.kg.get_node(finding_id))

            # Reject nodes that are not findings.
            is_finding = finding.get("type") == "finding"
            checks["is_finding"] = is_finding

            if not is_finding:
                return self._response(
                    finding_id,
                    checks,
                    reason="Node is not a finding.",
                )

            # Collect evidence and model relationships.
            evidence_ids = []
            model_ids = []
            invalid_evidence_targets = []
            invalid_model_targets = []

            for _, target, data in self.kg.graph.out_edges(
                finding_id, data=True
            ):
                relation = data.get("relation")

                if relation == "supported_by":
                    if (
                        target in self.kg.graph.nodes
                        and self.kg.get_node(target).get("type")
                        == "evidence"
                    ):
                        evidence_ids.append(target)
                    else:
                        invalid_evidence_targets.append(target)

                elif relation == "generated_by":
                    if (
                        target in self.kg.graph.nodes
                        and self.kg.get_node(target).get("type")
                        == "model"
                    ):
                        model_ids.append(target)
                    else:
                        invalid_model_targets.append(target)

            # Validate confidence, including NaN and infinity.
            confidence = finding.get("confidence")
            has_confidence = confidence is not None

            try:
                valid_confidence = (
                    isinstance(confidence, (int, float))
                    and not isinstance(confidence, bool)
                    and math.isfinite(confidence)
                    and 0.0 <= confidence <= 1.0
                )
            except (TypeError, ValueError, OverflowError):
                valid_confidence = False

            checks["has_confidence"] = has_confidence
            checks["valid_confidence"] = valid_confidence

            # Evidence must be connected to the finding and have
            # a non-empty textual description.
            has_evidence = bool(evidence_ids)

            evidence_connected = (
                has_evidence
                and not invalid_evidence_targets
            )

            valid_evidence_content = bool(evidence_ids) and all(
                isinstance(
                    self.kg.get_node(evidence_id).get("description"),
                    str,
                )
                and bool(
                    self.kg.get_node(evidence_id)
                    .get("description", "")
                    .strip()
                )
                for evidence_id in evidence_ids
            )

            checks["has_evidence"] = has_evidence
            checks["evidence_connected"] = evidence_connected
            checks["valid_evidence_content"] = valid_evidence_content

            # Models are optional, but connected models must be valid.
            has_model = bool(model_ids)
            valid_model_accuracy = not invalid_model_targets

            for model_id in model_ids:
                model = self.kg.get_node(model_id)
                accuracy = model.get("accuracy")

                # Missing accuracy is allowed. If present, it must be
                # numeric, finite, and within [0, 1].
                if accuracy is not None:
                    try:
                        valid_accuracy = (
                            isinstance(accuracy, (int, float))
                            and not isinstance(accuracy, bool)
                            and math.isfinite(accuracy)
                            and 0.0 <= accuracy <= 1.0
                        )
                    except (TypeError, ValueError, OverflowError):
                        valid_accuracy = False

                    if not valid_accuracy:
                        valid_model_accuracy = False
                        break

            checks["has_model"] = has_model
            checks["valid_model_accuracy"] = valid_model_accuracy

            # Determine structural validity only.
            if all([
                checks["has_confidence"],
                checks["valid_confidence"],
                checks["has_evidence"],
                checks["evidence_connected"],
                checks["valid_evidence_content"],
                checks["valid_model_accuracy"],
            ]):
                reason = (
                    "Structural checks passed. Scientific support for "
                    "the finding has not been assessed."
                )
            else:
                failed_checks = [
                    name
                    for name, passed in checks.items()
                    if name not in {"has_model"}
                    and not passed
                ]

                reason = (
                    "Structural verification failed: "
                    + ", ".join(failed_checks)
                    + "."
                )

            return self._response(
                finding_id,
                checks,
                reason=reason,
                confidence=confidence if has_confidence else 0.0,
                evidence=evidence_ids,
                models=model_ids,
            )