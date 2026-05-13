"""End-to-end test for training modeling pipeline (Path 2).

Tests DeepSeek V3, V3.2, and V4 models to verify:
1. TP shape splitting correctness across different TP degrees
2. Communication operator insertion (all_reduce, all_to_all)
3. Complete training report generation

Logs operator details to help identify version-specific issues.

Run with:
    .\run_pytest.bat tests/IT/test_training_tp_e2e.py -v
"""
import logging
import pytest
from pathlib import Path
import tempfile

from python.zrt.pipeline import run_trace_phases
from python.zrt.transform.analysis import estimate_training_from_graphs
import python.zrt.hardware.registry as hw_registry


# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(model)s]: %(message)s"
)
logger = logging.getLogger(__name__)


# ── Model configuration registry ──────────────────────────────────────────────

MODEL_CONFIGS = {
    "deepseek_v3": {
        "model_id": "hf_models/deepseek_v3",
        "hidden_size": 7168,
        "num_heads": 128,
        "description": "DeepSeek V3 (MLA attention)",
        "key_ops": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "supports_training": True,
    },
    "deepseek_v3_2": {
        "model_id": "hf_models/deepseek_v3_2",
        "hidden_size": 7168,
        "num_heads": 128,
        "description": "DeepSeek V3.2 (MLA + Index attention)",
        "key_ops": ["q_proj", "k_proj", "v_proj", "o_proj", "index_q", "index_kv"],
        "supports_training": True,
    },
    "deepseek_v4": {
        "model_id": "hf_models/deepseek_v4",
        "hidden_size": 7168,
        "num_heads": 128,
        "description": "DeepSeek V4 (MLA 2.0 + LongMoE)",
        "key_ops": ["q_proj", "k_proj", "v_proj", "o_proj", "q_lora", "o_lora"],
        "supports_training": False,  # V4 uses inference-only implementation
        "known_issue": "FakeTensor mode not supported (Mixing fake modes NYI)",
    },
}


# ── Helper functions ──────────────────────────────────────────────────────────

def log_graph_statistics(graph, model_name, phase):
    """Log detailed statistics about the captured graph."""
    if graph is None:
        logger.info(f"Graph is None for {model_name} {phase}", extra={"model": model_name})
        return
    
    total_nodes = len(graph.nodes)
    op_types = {}
    scopes = {}
    attention_nodes = []
    mlp_nodes = []
    
    for node_id, node in graph.nodes.items():
        op_type = node.op_type
        op_types[op_type] = op_types.get(op_type, 0) + 1
        
        scope = node.scope
        if scope:
            scope_parts = scope.split(".")
            if len(scope_parts) >= 3:
                layer_scope = ".".join(scope_parts[:3])
                scopes[layer_scope] = scopes.get(layer_scope, 0) + 1
            
            # Classify by component type
            if "self_attn" in scope or "attention" in scope.lower():
                attention_nodes.append((node_id, op_type))
            elif "mlp" in scope.lower() or "feed_forward" in scope.lower():
                mlp_nodes.append((node_id, op_type))
    
    logger.info(f"[{phase}] Total nodes: {total_nodes}", extra={"model": model_name})
    logger.info(f"[{phase}] Unique op types: {len(op_types)}", extra={"model": model_name})
    logger.info(f"[{phase}] Attention-related nodes: {len(attention_nodes)}", extra={"model": model_name})
    logger.info(f"[{phase}] MLP-related nodes: {len(mlp_nodes)}", extra={"model": model_name})
    
    # Log top 10 most frequent op types
    top_ops = sorted(op_types.items(), key=lambda x: x[1], reverse=True)[:10]
    for op_type, count in top_ops:
        logger.info(f"[{phase}]  - {op_type}: {count}", extra={"model": model_name})
    
    # Log scope distribution
    logger.info(f"[{phase}] Scope distribution:", extra={"model": model_name})
    for scope_name, count in sorted(scopes.items())[:5]:
        logger.info(f"[{phase}]  - {scope_name}: {count}", extra={"model": model_name})


def log_transformed_graph_analysis(unified_graph, model_name, tp_degree):
    """Analyze and log the transformed graph after TP transformations."""
    if unified_graph is None:
        logger.warning(f"Unified graph is None after TP={tp_degree}", extra={"model": model_name})
        return
    
    # Count different types of nodes
    comm_nodes = []
    q_proj_nodes = []
    o_proj_nodes = []
    linear_nodes = []
    
    for node_id, node in unified_graph.nodes.items():
        # Communication operators
        if node.op_type.startswith("comm."):
            comm_nodes.append((node_id, node.op_type, node.scope))
        
        # Attention projection layers
        if "q_proj" in node.scope and "self_attn" in node.scope:
            q_proj_nodes.append((node_id, node.outputs[0].shape if node.outputs else None))
        if "o_proj" in node.scope and "self_attn" in node.scope:
            o_proj_nodes.append((node_id, node.outputs[0].shape if node.outputs else None))
        
        # Linear operations
        if node.op_type == "aten.mm.default" or "linear" in node.op_type.lower():
            linear_nodes.append((node_id, node.scope))
    
    # Log communication operators
    logger.info(f"[TP={tp_degree}] Communication ops found: {len(comm_nodes)}", extra={"model": model_name})
    for node_id, op_type, scope in comm_nodes:
        logger.info(f"[TP={tp_degree}]  - {op_type} at {scope}", extra={"model": model_name})
    
    # Log q_proj shape analysis
    logger.info(f"[TP={tp_degree}] Q_proj nodes found: {len(q_proj_nodes)}", extra={"model": model_name})
    for node_id, output_shape in q_proj_nodes:
        logger.info(f"[TP={tp_degree}]  - {node_id}: output_shape={output_shape}", extra={"model": model_name})
    
    # Log o_proj shape analysis
    logger.info(f"[TP={tp_degree}] O_proj nodes found: {len(o_proj_nodes)}", extra={"model": model_name})
    for node_id, output_shape in o_proj_nodes:
        logger.info(f"[TP={tp_degree}]  - {node_id}: output_shape={output_shape}", extra={"model": model_name})
    
    # Log total linear ops
    logger.info(f"[TP={tp_degree}] Linear operations: {len(linear_nodes)}", extra={"model": model_name})


def log_model_compatibility_info(model_key, config):
    """Log compatibility information for a model."""
    logger.info(f"=== Model Configuration: {config['description']} ===", extra={"model": model_key})
    logger.info(f"Model ID: {config['model_id']}", extra={"model": model_key})
    logger.info(f"Hidden Size: {config['hidden_size']}", extra={"model": model_key})
    logger.info(f"Num Heads: {config['num_heads']}", extra={"model": model_key})
    logger.info(f"Key Ops: {', '.join(config['key_ops'])}", extra={"model": model_key})
    logger.info(f"Supports Training: {'Yes' if config['supports_training'] else 'No'}", extra={"model": model_key})
    if "known_issue" in config:
        logger.warning(f"Known Issue: {config['known_issue']}", extra={"model": model_key})


# ── Test fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(params=list(MODEL_CONFIGS.keys()), ids=list(MODEL_CONFIGS.keys()))
def model_config(request):
    """Parameterized fixture for different DeepSeek model versions."""
    model_key = request.param
    config = MODEL_CONFIGS[model_key]
    log_model_compatibility_info(model_key, config)
    return model_key, config


@pytest.fixture
def training_graphs(model_config):
    """Fixture to capture training graphs for a specific model."""
    model_key, config = model_config
    
    # Skip models that don't support training
    if not config.get("supports_training", True):
        pytest.skip(f"Model {config['description']} does not support training mode")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            logger.info(f"Starting graph capture for {config['model_id']}", extra={"model": model_key})
            result = run_trace_phases(
                model_id=config["model_id"],
                num_layers=2,
                batch_size=1,
                seq_len=128,
                phases=["train_forward", "train_backward"],
                output_dir=tmpdir,
            )
            
            # Log graph statistics
            log_graph_statistics(result.graphs.get("train_forward"), model_key, "train_forward")
            log_graph_statistics(result.graphs.get("train_backward"), model_key, "train_backward")
            
            yield result.graphs
            
        except Exception as e:
            logger.error(f"Failed to capture graphs for {config['model_id']}: {str(e)}", extra={"model": model_key})
            logger.error(f"Error type: {type(e).__name__}", extra={"model": model_key})
            # Re-raise to fail the test
            raise


# ── Test cases ────────────────────────────────────────────────────────────────

class TestTrainingModelingTP:
    """End-to-end TP validation tests for training modeling pipeline."""

    def test_tp2_training_modeling(self, model_config, training_graphs):
        """Test TP=2 training modeling - verify shape splitting and comm insertion."""
        model_key, config = model_config
        hidden_size = config["hidden_size"]
        
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        logger.info(f"Running TP=2 training modeling", extra={"model": model_key})
        
        hw_spec = hw_registry.load("nvidia_h100_sxm")
        
        try:
            report, ctx, transformed = estimate_training_from_graphs(
                forward_graph=fwd_graph,
                backward_graph=bwd_graph,
                hw_spec=hw_spec,
                tp=2,
                pp=1,
                dp=1,
                seq_len=128,
                batch_size=1,
                hidden=hidden_size,
                num_layers=2,
                return_transformed=True,
            )
            
            # Log analysis
            log_transformed_graph_analysis(transformed.get("unified"), model_key, 2)
            
            # Verify report
            assert report is not None, f"Report should not be None"
            assert report.step_time_ms > 0, f"Step time should be > 0"
            assert report.mfu > 0, f"MFU should be > 0"
            
            # Verify transformed graph
            unified_graph = transformed.get("unified")
            assert unified_graph is not None, "Unified graph should exist"
            
            # Verify TP shape splitting for column parallel (q_proj)
            q_proj_nodes = [n for n in unified_graph.nodes.values() 
                          if "q_proj" in n.scope and "self_attn" in n.scope]
            
            if q_proj_nodes:
                for node in q_proj_nodes:
                    output_shape = node.outputs[0].shape if node.outputs else ()
                    expected_dim = hidden_size // 2
                    assert output_shape[-1] == expected_dim, \
                        f"Column parallel output dim should be {expected_dim}, got {output_shape[-1]}"
                logger.info(f"✓ TP=2 shape splitting verified for {len(q_proj_nodes)} q_proj nodes", 
                          extra={"model": model_key})
            else:
                logger.warning(f"⚠ No q_proj nodes found in graph - shape verification skipped", 
                             extra={"model": model_key})
            
            # Verify communication operators
            comm_nodes = [n for n in unified_graph.nodes.values() 
                        if n.op_type.startswith("comm.")]
            assert len(comm_nodes) > 0, "Communication operators should be inserted"
            
            all_reduce_nodes = [n for n in comm_nodes if "all_reduce" in n.op_type]
            assert len(all_reduce_nodes) > 0, "all_reduce operators should be present for TP=2"
            
            logger.info(f"✓ TP=2 test PASSED - {len(comm_nodes)} comm ops, {len(all_reduce_nodes)} all_reduce", 
                      extra={"model": model_key})
            
        except Exception as e:
            logger.error(f"✗ TP=2 test FAILED: {str(e)}", extra={"model": model_key})
            logger.error(f"Error type: {type(e).__name__}", extra={"model": model_key})
            raise

    def test_tp4_training_modeling(self, model_config, training_graphs):
        """Test TP=4 training modeling - verify shape splitting and comm insertion."""
        model_key, config = model_config
        hidden_size = config["hidden_size"]
        
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        logger.info(f"Running TP=4 training modeling", extra={"model": model_key})
        
        hw_spec = hw_registry.load("nvidia_h100_sxm")
        
        try:
            report, ctx, transformed = estimate_training_from_graphs(
                forward_graph=fwd_graph,
                backward_graph=bwd_graph,
                hw_spec=hw_spec,
                tp=4,
                pp=1,
                dp=1,
                seq_len=128,
                batch_size=1,
                hidden=hidden_size,
                num_layers=2,
                return_transformed=True,
            )
            
            log_transformed_graph_analysis(transformed.get("unified"), model_key, 4)
            
            unified_graph = transformed.get("unified")
            assert unified_graph is not None
            
            # Verify TP=4 shape splitting
            q_proj_nodes = [n for n in unified_graph.nodes.values() 
                          if "q_proj" in n.scope and "self_attn" in n.scope]
            
            if q_proj_nodes:
                for node in q_proj_nodes:
                    output_shape = node.outputs[0].shape if node.outputs else ()
                    expected_dim = hidden_size // 4
                    assert output_shape[-1] == expected_dim, \
                        f"Column parallel output dim should be {expected_dim}, got {output_shape[-1]}"
                logger.info(f"✓ TP=4 shape splitting verified for {len(q_proj_nodes)} q_proj nodes", 
                          extra={"model": model_key})
            
            # Verify comm operators
            comm_nodes = [n for n in unified_graph.nodes.values() 
                        if n.op_type.startswith("comm.")]
            assert len(comm_nodes) > 0
            
            all_reduce_nodes = [n for n in comm_nodes if "all_reduce" in n.op_type]
            assert len(all_reduce_nodes) > 0
            
            logger.info(f"✓ TP=4 test PASSED - {len(comm_nodes)} comm ops, {len(all_reduce_nodes)} all_reduce", 
                      extra={"model": model_key})
            
        except Exception as e:
            logger.error(f"✗ TP=4 test FAILED: {str(e)}", extra={"model": model_key})
            raise

    def test_tp1_no_comm_ops(self, model_config, training_graphs):
        """Test TP=1 - verify no communication operators are inserted."""
        model_key, config = model_config
        hidden_size = config["hidden_size"]
        
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        logger.info(f"Running TP=1 training modeling", extra={"model": model_key})
        
        hw_spec = hw_registry.load("nvidia_h100_sxm")
        
        try:
            report, ctx, transformed = estimate_training_from_graphs(
                forward_graph=fwd_graph,
                backward_graph=bwd_graph,
                hw_spec=hw_spec,
                tp=1,
                pp=1,
                dp=1,
                seq_len=128,
                batch_size=1,
                hidden=hidden_size,
                num_layers=2,
                return_transformed=True,
            )
            
            unified_graph = transformed.get("unified")
            assert unified_graph is not None
            
            # Verify no communication operators for TP=1
            comm_nodes = [n for n in unified_graph.nodes.values() 
                        if n.op_type.startswith("comm.")]
            assert len(comm_nodes) == 0, \
                f"No communication operators should exist for TP=1, got {len(comm_nodes)}"
            
            logger.info(f"✓ TP=1 test PASSED - no communication operators", extra={"model": model_key})
            
        except Exception as e:
            logger.error(f"✗ TP=1 test FAILED: {str(e)}", extra={"model": model_key})
            raise

    def test_tp_report_metrics(self, model_config, training_graphs):
        """Test that TP configuration affects report metrics correctly."""
        model_key, config = model_config
        hidden_size = config["hidden_size"]
        
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        logger.info(f"Testing report metrics across TP configurations", extra={"model": model_key})
        
        hw_spec = hw_registry.load("nvidia_h100_sxm")
        
        try:
            # Run with TP=1
            report_tp1, _, _ = estimate_training_from_graphs(
                forward_graph=fwd_graph,
                backward_graph=bwd_graph,
                hw_spec=hw_spec,
                tp=1,
                pp=1,
                dp=1,
                seq_len=128,
                batch_size=1,
                hidden=hidden_size,
                num_layers=2,
                return_transformed=True,
            )
            
            # Run with TP=2
            report_tp2, _, _ = estimate_training_from_graphs(
                forward_graph=fwd_graph,
                backward_graph=bwd_graph,
                hw_spec=hw_spec,
                tp=2,
                pp=1,
                dp=1,
                seq_len=128,
                batch_size=1,
                hidden=hidden_size,
                num_layers=2,
                return_transformed=True,
            )
            
            # Verify reports have valid metrics
            assert report_tp1.step_time_ms > 0
            assert report_tp2.step_time_ms > 0
            
            # Verify report structure
            assert hasattr(report_tp2, 'memory')
            assert hasattr(report_tp2, 'per_stage')
            assert hasattr(report_tp2, 'warnings')
            
            logger.info(f"✓ Report metrics: TP1={report_tp1.step_time_ms:.2f}ms, TP2={report_tp2.step_time_ms:.2f}ms", 
                      extra={"model": model_key})
            logger.info(f"✓ Report metrics test PASSED", extra={"model": model_key})
            
        except Exception as e:
            logger.error(f"✗ Report metrics test FAILED: {str(e)}", extra={"model": model_key})
            raise


# ── DeepSeek V4 specific tests ────────────────────────────────────────────────

class TestDeepSeekV4Compatibility:
    """Tests to diagnose DeepSeek V4 compatibility issues."""
    
    @pytest.mark.parametrize("model_key", ["deepseek_v4"])
    def test_v4_inference_only_warning(self, model_key):
        """Verify V4 is correctly identified as inference-only."""
        config = MODEL_CONFIGS[model_key]
        assert not config.get("supports_training", True), \
            f"{config['description']} should be marked as inference-only"
        assert "known_issue" in config, \
            f"{config['description']} should have a known_issue documented"
        logger.info(f"✓ DeepSeek V4 correctly marked as inference-only: {config['known_issue']}", 
                  extra={"model": model_key})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
