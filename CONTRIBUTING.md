# Contributing to SAR Water Detection Lab

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🎯 How to Contribute

### Reporting Bugs

1. **Check existing issues** first to avoid duplicates
2. **Use the issue template** with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version)
   - Screenshots if applicable

### Suggesting Features

1. **Open an issue** with the "Feature Request" label
2. **Describe the feature** clearly:
   - What problem does it solve?
   - How would it work?
   - Any implementation ideas?

### Contributing Code

#### 1. Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/sar-water-detection-lab.git
cd sar-water-detection-lab/chips/gui

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (including dev tools)
pip install -r requirements.txt
pip install black flake8 pytest mypy

# Create a feature branch
git checkout -b feature/your-feature-name
```

#### 2. Code Standards

**Python Style**
- Follow PEP 8 guidelines
- Use type hints where possible
- Maximum line length: 100 characters
- Use meaningful variable names

```python
# Good
def calculate_water_percentage(mask: np.ndarray) -> float:
    """Calculate percentage of water pixels in binary mask."""
    return (mask.sum() / mask.size) * 100

# Avoid
def calc(m):
    return m.sum()/m.size*100
```

**Formatting**
```bash
# Format your code before committing
black *.py

# Check for style issues
flake8 *.py --max-line-length=100

# Type checking
mypy app.py --ignore-missing-imports
```

**Documentation**
- Add docstrings to all functions and classes
- Use Google-style docstrings
- Update README.md if adding features

```python
def new_filter(data: np.ndarray, threshold: float = -17.0) -> np.ndarray:
    """
    Apply a new water detection filter.
    
    Args:
        data: Input SAR data in dB
        threshold: Detection threshold in dB (default: -17.0)
        
    Returns:
        Binary mask where True indicates water
        
    Example:
        >>> mask = new_filter(vh_data, threshold=-19.0)
    """
    return data < threshold
```

#### 3. Adding New Filters

To add a new water detection algorithm:

1. **Add implementation** to `filter_engine_complete.py`
2. **Add filter spec** to `app.py` FILTER_SPECS dict
3. **Add filter call** in `apply_filter()` function
4. **Test thoroughly** on sample data
5. **Document** the algorithm and parameters

Example:

```python
# In filter_engine_complete.py
def my_new_filter(data, param1=10, param2=0.5):
    """
    My new water detection filter.
    
    Based on: Paper et al. (2024)
    """
    # Implementation
    result = ...
    return result

# In app.py FILTER_SPECS
'My New Filter': {
    'param1': {'type': 'slider', 'min': 1, 'max': 20, 'default': 10, 'step': 1},
    'param2': {'type': 'slider', 'min': 0.1, 'max': 1.0, 'default': 0.5, 'step': 0.1}
}

# In app.py apply_filter()
elif filter_name == 'My New Filter':
    return fe.my_new_filter(
        chip_data['vh'],
        param1=params.get('param1', 10),
        param2=params.get('param2', 0.5)
    )
```

#### 4. Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Test your changes manually
streamlit run app.py
```

#### 5. Commit Guidelines

Use conventional commits:

```
feat: add Gaussian mixture model filter
fix: resolve NaN handling in HAND processing
docs: update installation instructions
refactor: simplify threshold calculation
test: add unit tests for texture features
```

#### 6. Submit Pull Request

1. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open PR** on GitHub with:
   - Clear title describing the change
   - Detailed description of what/why/how
   - Reference related issues
   - Screenshots for UI changes

3. **Review process**
   - Maintainers will review your code
   - Address feedback promptly
   - Keep PR focused (one feature per PR)

## 🏗️ Architecture Overview

```
app.py                      # Main Streamlit UI
├── config.py               # Configuration management
├── filter_engine_complete.py  # 47+ water detection algorithms
├── analysis_module.py      # Analysis utilities
├── qa_module.py           # Quality assurance
└── presets.py             # Pre-configured filter sets
```

**Key Components:**

- **Filter Engine**: Pure functions that accept data and return masks
- **App Logic**: Streamlit UI handling and session state
- **Config System**: Environment-based configuration
- **QA System**: Quality tracking and audit trails

## 🐛 Debugging Tips

```python
# Enable debug logging in Streamlit
import logging
logging.basicConfig(level=logging.DEBUG)

# Inspect session state
st.write(st.session_state)

# Profile performance
import cProfile
cProfile.run('apply_filter(...)')
```

## 📝 Documentation

- **Code comments**: Explain complex logic
- **Docstrings**: All public functions/classes
- **README updates**: New features or breaking changes
- **Examples**: Add usage examples for new features

## ❓ Questions?

- **Open an issue** for technical questions
- **Join discussions** for design discussions
- **Email maintainers** for security issues (do not open public issues)

## 📜 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on what's best for the community

## 🙏 Recognition

Contributors will be:
- Listed in README.md
- Credited in release notes
- Eligible for "Contributor" badge

---

Thank you for making SAR Water Detection Lab better! 🌊
