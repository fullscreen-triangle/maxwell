# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email the maintainer directly at: kundai.sachikonye@wzw.tum.de
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

You can expect:
- Acknowledgment within 48 hours
- Status update within 7 days
- Resolution timeline based on severity

## Security Considerations

This is a scientific computing project. Key security considerations:

### Numerical Stability
- Floating-point operations may have precision limits
- Large networks may cause memory issues
- Recursive operations are depth-limited

### Input Validation
- CLI inputs are validated via `clap`
- Configuration parameters are bounds-checked
- File paths are sanitized

### Dependencies
- Dependencies are regularly audited with `cargo audit`
- Minimal dependency footprint
- No network operations in core library

## Best Practices

When using this library:

1. **Validate inputs** before passing to the processor
2. **Limit recursion depth** for large systems
3. **Monitor memory usage** for large networks
4. **Use release builds** for production

