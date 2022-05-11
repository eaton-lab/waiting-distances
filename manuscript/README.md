


When editing:
- try to comment old writing to preserve it in draft stage.


To rebuild PDF run:
```bash
pdflatex waiting-distances
pdflatex waiting-distances
```

If citations change run:
```bash
bibtex waiting-distances
bibtex waiting-distances
```

To push changes:
- Your changes should be in `waiting-distances.tex` and `references.bib`
- Other file changes should remain local to your repo. Gitignore file should take care of this.
- If you want to push a pdf, make a copy renamed draft-X.pdf and push.
- This is so that we do not need to merge/notmerge pdf conflicts.

