"""
Tests for the dissolution validation module.
"""

import pytest
from maxwell_validation.dissolution import DissolutionValidator, DissolutionArgument


class TestDissolutionValidator:
    """Test suite for the DissolutionValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create a fresh validator for each test."""
        return DissolutionValidator()
    
    def test_temporal_triviality(self, validator):
        """Test that temporal triviality is validated."""
        result = validator.validate_temporal_triviality()
        assert result.validated, result.message
        assert result.argument == DissolutionArgument.TEMPORAL_TRIVIALITY
    
    def test_phase_lock_temperature_independence(self, validator):
        """Test that phase-lock networks are temperature-independent."""
        result = validator.validate_phase_lock_temperature_independence()
        assert result.validated, result.message
        assert result.argument == DissolutionArgument.PHASE_LOCK_TEMPERATURE_INDEPENDENCE
    
    def test_retrieval_paradox(self, validator):
        """Test the retrieval paradox demonstration."""
        result = validator.validate_retrieval_paradox()
        assert result.validated, result.message
        assert result.argument == DissolutionArgument.RETRIEVAL_PARADOX
    
    def test_dissolution_of_observation(self, validator):
        """Test that observation is unnecessary."""
        result = validator.validate_dissolution_of_observation()
        assert result.validated, result.message
        assert result.argument == DissolutionArgument.DISSOLUTION_OF_OBSERVATION
    
    def test_dissolution_of_decision(self, validator):
        """Test that decision-making is unnecessary."""
        result = validator.validate_dissolution_of_decision()
        assert result.validated, result.message
        assert result.argument == DissolutionArgument.DISSOLUTION_OF_DECISION
    
    def test_dissolution_of_second_law(self, validator):
        """Test that entropy increases."""
        result = validator.validate_dissolution_of_second_law()
        assert result.validated, result.message
        assert result.argument == DissolutionArgument.DISSOLUTION_OF_SECOND_LAW
    
    def test_information_complementarity(self, validator):
        """Test information complementarity."""
        result = validator.validate_information_complementarity()
        assert result.validated, result.message
        assert result.argument == DissolutionArgument.INFORMATION_COMPLEMENTARITY
    
    def test_all_seven_arguments(self, validator):
        """Test that all seven arguments are validated."""
        results = validator.run_all_validations()
        
        assert len(results) == 7, "Should have exactly 7 dissolution arguments"
        
        for name, result in results.items():
            assert result.validated, f"Argument {name} failed: {result.message}"
    
    def test_no_demon(self, validator):
        """The ultimate test: there is no demon."""
        results = validator.run_all_validations()
        all_passed = all(r.validated for r in results.values())
        
        assert all_passed, "THERE IS NO DEMON - All seven arguments must pass"


class TestDissolutionArguments:
    """Test the dissolution argument enum."""
    
    def test_argument_count(self):
        """Ensure we have exactly 7 arguments."""
        arguments = list(DissolutionArgument)
        assert len(arguments) == 7
    
    def test_argument_values(self):
        """Ensure arguments are numbered 1-7."""
        for i, arg in enumerate(DissolutionArgument, start=1):
            assert arg.value == i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

