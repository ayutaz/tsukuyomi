"""
Tests for MOS (Mean Opinion Score) evaluation tools
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import json

from src.tools.mos_evaluation import (
    MOSEvaluator,
    MOSCollector,
    MOSAnalyzer,
    ABTestEvaluator,
    AutomaticMOSPredictor,
    EvaluationConfig,
    create_evaluation_webapp
)


class TestMOSEvaluator:
    """Test MOS evaluator"""
    
    def test_evaluator_creation(self):
        """Test MOS evaluator creation"""
        config = EvaluationConfig()
        evaluator = MOSEvaluator(config)
        
        assert evaluator.config == config
        assert evaluator.results == []
    
    def test_add_evaluation(self):
        """Test adding evaluation"""
        evaluator = MOSEvaluator()
        
        # Add evaluation
        evaluator.add_evaluation(
            audio_id="test_001",
            score=4.5,
            listener_id="listener_001",
            metadata={
                "speaker": "speaker_001",
                "emotion": "neutral"
            }
        )
        
        assert len(evaluator.results) == 1
        assert evaluator.results[0]['audio_id'] == "test_001"
        assert evaluator.results[0]['score'] == 4.5
    
    def test_calculate_mos(self):
        """Test MOS calculation"""
        evaluator = MOSEvaluator()
        
        # Add multiple evaluations
        for i in range(5):
            evaluator.add_evaluation(
                audio_id="test_001",
                score=4.0 + i * 0.2,
                listener_id=f"listener_{i}"
            )
        
        # Calculate MOS
        mos = evaluator.calculate_mos("test_001")
        
        assert mos['mean'] == pytest.approx(4.4, 0.01)
        assert mos['std'] > 0
        assert mos['count'] == 5
        assert mos['confidence_interval'] is not None
    
    def test_save_load_results(self):
        """Test saving and loading results"""
        evaluator = MOSEvaluator()
        
        # Add evaluations
        evaluator.add_evaluation("test_001", 4.5, "listener_001")
        evaluator.add_evaluation("test_002", 3.8, "listener_001")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save results
            save_path = Path(tmpdir) / "results.json"
            evaluator.save_results(save_path)
            
            assert save_path.exists()
            
            # Load results
            new_evaluator = MOSEvaluator()
            new_evaluator.load_results(save_path)
            
            assert len(new_evaluator.results) == 2
            assert new_evaluator.results[0]['audio_id'] == "test_001"


class TestMOSCollector:
    """Test MOS collection interface"""
    
    def test_collector_creation(self):
        """Test MOS collector creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MOSCollector(
                audio_dir=tmpdir,
                output_file="mos_results.json"
            )
            
            assert collector.audio_dir == Path(tmpdir)
            assert collector.output_file == "mos_results.json"
            assert collector.current_index == 0
    
    def test_get_audio_files(self):
        """Test getting audio files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy audio files
            audio_dir = Path(tmpdir)
            (audio_dir / "audio1.wav").touch()
            (audio_dir / "audio2.wav").touch()
            (audio_dir / "text.txt").touch()  # Non-audio file
            
            collector = MOSCollector(audio_dir=tmpdir)
            files = collector.get_audio_files()
            
            assert len(files) == 2
            assert all(f.suffix == '.wav' for f in files)
    
    def test_collect_score(self):
        """Test score collection"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy audio
            audio_file = Path(tmpdir) / "test.wav"
            audio_file.touch()
            
            collector = MOSCollector(audio_dir=tmpdir)
            
            # Collect score
            collector.collect_score(
                audio_file=audio_file,
                score=4.2,
                listener_id="test_listener"
            )
            
            assert len(collector.evaluator.results) == 1
            assert collector.evaluator.results[0]['score'] == 4.2


class TestMOSAnalyzer:
    """Test MOS analysis tools"""
    
    def test_analyzer_creation(self):
        """Test MOS analyzer creation"""
        # Create dummy results
        results = [
            {'audio_id': 'audio1', 'score': 4.5, 'listener_id': 'l1', 'metadata': {'speaker': 's1'}},
            {'audio_id': 'audio1', 'score': 4.0, 'listener_id': 'l2', 'metadata': {'speaker': 's1'}},
            {'audio_id': 'audio2', 'score': 3.5, 'listener_id': 'l1', 'metadata': {'speaker': 's2'}},
        ]
        
        analyzer = MOSAnalyzer(results)
        
        assert len(analyzer.results) == 3
        assert analyzer.df is not None
        assert len(analyzer.df) == 3
    
    def test_overall_statistics(self):
        """Test overall statistics calculation"""
        results = [
            {'audio_id': f'audio{i}', 'score': 3.0 + i * 0.5, 'listener_id': 'l1'}
            for i in range(5)
        ]
        
        analyzer = MOSAnalyzer(results)
        stats = analyzer.get_overall_statistics()
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'median' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert stats['mean'] == pytest.approx(4.0, 0.1)
    
    def test_speaker_statistics(self):
        """Test per-speaker statistics"""
        results = [
            {'audio_id': 'a1', 'score': 4.5, 'listener_id': 'l1', 'metadata': {'speaker': 's1'}},
            {'audio_id': 'a2', 'score': 4.0, 'listener_id': 'l1', 'metadata': {'speaker': 's1'}},
            {'audio_id': 'a3', 'score': 3.5, 'listener_id': 'l1', 'metadata': {'speaker': 's2'}},
            {'audio_id': 'a4', 'score': 3.0, 'listener_id': 'l1', 'metadata': {'speaker': 's2'}},
        ]
        
        analyzer = MOSAnalyzer(results)
        speaker_stats = analyzer.get_speaker_statistics()
        
        assert 's1' in speaker_stats
        assert 's2' in speaker_stats
        assert speaker_stats['s1']['mean'] == pytest.approx(4.25, 0.01)
        assert speaker_stats['s2']['mean'] == pytest.approx(3.25, 0.01)
    
    def test_listener_agreement(self):
        """Test inter-rater agreement calculation"""
        results = [
            {'audio_id': 'a1', 'score': 4.0, 'listener_id': 'l1'},
            {'audio_id': 'a1', 'score': 4.5, 'listener_id': 'l2'},
            {'audio_id': 'a2', 'score': 3.0, 'listener_id': 'l1'},
            {'audio_id': 'a2', 'score': 3.5, 'listener_id': 'l2'},
        ]
        
        analyzer = MOSAnalyzer(results)
        agreement = analyzer.calculate_listener_agreement()
        
        assert 'correlation' in agreement
        assert agreement['correlation'] > 0  # Should be positive
    
    def test_plot_generation(self):
        """Test plot generation"""
        results = [
            {'audio_id': f'audio{i}', 'score': 3.0 + i * 0.3, 'listener_id': 'l1'}
            for i in range(10)
        ]
        
        analyzer = MOSAnalyzer(results)
        
        with patch('matplotlib.pyplot.savefig') as mock_save:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                analyzer.generate_plots(output_dir)
                
                # Check plots were saved
                assert mock_save.call_count >= 2  # At least distribution and per-audio


class TestABTestEvaluator:
    """Test A/B test evaluator"""
    
    def test_ab_test_creation(self):
        """Test A/B test evaluator creation"""
        evaluator = ABTestEvaluator()
        
        assert evaluator.comparisons == []
        assert evaluator.systems == set()
    
    def test_add_comparison(self):
        """Test adding comparison"""
        evaluator = ABTestEvaluator()
        
        evaluator.add_comparison(
            audio_id="test_001",
            system_a="baseline",
            system_b="proposed",
            preference="proposed",
            listener_id="listener_001"
        )
        
        assert len(evaluator.comparisons) == 1
        assert "baseline" in evaluator.systems
        assert "proposed" in evaluator.systems
    
    def test_calculate_preference_scores(self):
        """Test preference score calculation"""
        evaluator = ABTestEvaluator()
        
        # Add comparisons
        for i in range(10):
            preference = "proposed" if i < 7 else "baseline"
            evaluator.add_comparison(
                audio_id=f"test_{i}",
                system_a="baseline",
                system_b="proposed",
                preference=preference,
                listener_id=f"listener_{i}"
            )
        
        scores = evaluator.calculate_preference_scores()
        
        assert scores["proposed"] == 0.7
        assert scores["baseline"] == 0.3
    
    def test_statistical_significance(self):
        """Test statistical significance testing"""
        evaluator = ABTestEvaluator()
        
        # Add comparisons with clear preference
        for i in range(20):
            preference = "proposed" if i < 15 else "baseline"
            evaluator.add_comparison(
                audio_id=f"test_{i}",
                system_a="baseline",
                system_b="proposed",
                preference=preference,
                listener_id=f"listener_{i}"
            )
        
        result = evaluator.test_significance("baseline", "proposed")
        
        assert 'p_value' in result
        assert 'is_significant' in result
        assert result['wins_a'] == 5
        assert result['wins_b'] == 15


class TestAutomaticMOSPredictor:
    """Test automatic MOS prediction"""
    
    def test_predictor_creation(self):
        """Test automatic MOS predictor creation"""
        predictor = AutomaticMOSPredictor(model_name="default")
        
        assert predictor.model_name == "default"
        assert predictor.model is None  # Not loaded yet
    
    @patch('torch.load')
    def test_model_loading(self, mock_load):
        """Test model loading"""
        # Mock model
        mock_model = Mock()
        mock_load.return_value = {'model_state_dict': {}}
        
        predictor = AutomaticMOSPredictor()
        
        with patch.object(predictor, '_create_model', return_value=mock_model):
            predictor.load_model("fake_path.pt")
            
            assert predictor.model is not None
    
    def test_predict_mos(self):
        """Test MOS prediction"""
        predictor = AutomaticMOSPredictor()
        
        # Mock model
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.return_value = torch.tensor([[4.2]])
        predictor.model = mock_model
        
        # Create dummy audio
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
        
        # Predict
        mos = predictor.predict(audio, sample_rate=22050)
        
        assert isinstance(mos, float)
        assert 1.0 <= mos <= 5.0
    
    def test_batch_prediction(self):
        """Test batch MOS prediction"""
        predictor = AutomaticMOSPredictor()
        
        # Mock model
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.return_value = torch.tensor([[4.2], [3.8], [4.5]])
        predictor.model = mock_model
        
        # Create dummy audios
        audios = [
            np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050)),
            np.sin(2 * np.pi * 880 * np.linspace(0, 1, 22050)),
            np.sin(2 * np.pi * 1320 * np.linspace(0, 1, 22050))
        ]
        
        # Predict batch
        scores = predictor.predict_batch(audios, sample_rate=22050)
        
        assert len(scores) == 3
        assert all(1.0 <= s <= 5.0 for s in scores)


class TestEvaluationWebApp:
    """Test evaluation web application"""
    
    @patch('streamlit.title')
    @patch('streamlit.sidebar')
    def test_webapp_creation(self, mock_sidebar, mock_title):
        """Test web app creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_evaluation_webapp(audio_dir=tmpdir)
            
            assert app is not None
            
            # Test app can be called
            app()
            
            mock_title.assert_called()


class TestEvaluationConfig:
    """Test evaluation configuration"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = EvaluationConfig()
        
        assert config.scale_min == 1
        assert config.scale_max == 5
        assert config.scale_step == 0.5
        assert config.require_listener_id == True
        assert config.randomize_order == True
        assert config.allow_replay == True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = EvaluationConfig(
            scale_min=0,
            scale_max=10,
            scale_step=1,
            require_listener_id=False
        )
        
        assert config.scale_min == 0
        assert config.scale_max == 10
        assert config.scale_step == 1
        assert config.require_listener_id == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])