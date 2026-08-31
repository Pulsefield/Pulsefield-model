# Documentation authoring

This guide owns repository-wide conventions for hand-authored Markdown.
Content-specific requirements remain with the document or subsystem that owns
the underlying contract.

## Markdown math

Delimit inline LaTeX with `$...$` and display LaTeX with `$$...$$`. Put each
display delimiter on its own line, with blank lines around the block:

```markdown
$$
f(x)=x^2
$$
```

Do not use `\(...\)` or `\[...\]`; renderer support for those delimiters is
inconsistent. When showing LaTeX source rather than rendering it, use an
inline code span or fenced code block.
