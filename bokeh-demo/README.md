# Visualization tool pilot project

1. Install by running
   ```
   poetry install
   ```
   for bare dependencies or
   ```
   poetry install --with examples
   ```
   to be able to run the example sites.

2. To serve the page, run
   ```
   bokeh serve examples/pilot examples/landing_page
   ```
   from a Poetry environment.


## Tailwind

We use tailwind for our CSS.
Install Tailwind CLI with
```bash
npm install -D tailwindcss
```
and then
```bash
npx tailwindcss -i examples/pilot/static/style_tailwind.css -o examples/pilot/static/style.css
```
to build new CSS.
