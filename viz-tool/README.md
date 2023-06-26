# Visualization tool pilot project

1. Install by running
   ```
   poetry install
   ```

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
npx tailwindcss -i examples/landing_page/static/style_tailwind.css -o examples/landing_page/static/style.css
```
to build new CSS.

## Docker

The app comes as a Docker app.
The data files, either a `DataManager` parquet cache or raw CSV files, is expected to be in `/mnt`.
The app is by default served on port 5006.

TODO: fix env vars
```bash
docker run --rm -p 5006:5006 -v $(readlink -f path/to/data):/mnt viz-tool
```
