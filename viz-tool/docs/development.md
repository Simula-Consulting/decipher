# Development

## Installation and Deployment :rocket:

!!! tip ""

    For information on how to run on TSD, see [the Apptainer section below](#apptainer-formerly-singularity-support).

### ... with Poetry

We use [Poetry](https://python-poetry.org/) to handle our project and dependencies.
To get started, run
```bash
poetry install
```
to set up the environment, and then
```bash
poetry run bokeh serve apps/pilot apps/landing_page
```
to launch the application.

### ... with Docker

The fastest way to get started, is to use docker compose:
```bash
# From viz-tool base folder
docker compose up
```

#### Apptainer (formerly Singularity) support

For running on [TSD](https://www.uio.no/tjenester/it/forskning/sensitiv/) we use [Apptainer](https://apptainer.org/).[^1]
To convert the Docker image to Apptainer format, run
```bash
docker run \
-v /var/run/docker.sock:/var/run/docker.sock \
-v ~/Downloads:/output \  # (1)!
-v /tmp/build_singularity:/tmp \  # (2)!
--privileged -t --rm \
quay.io/singularity/docker2singularity \
viz-tool
```

1. Sets the output of the generated image to `~/Downloads`
2. May not be needed. In some cases the default cache volume runs out of space.

??? note inline end "Using Apptainer to build"

    The above approach was chosen as it works well on any system.
    If you have to Apptainer on your local x86 machine, feel free to build directly
    with Apptainer.

where `viz-tool` is the name of the Docker image.
This will create a `.sif` file in your Downloads folder.

??? warning "Access to Apptainer"

    On TSD, Apptainer is not installed on the VMs by default.
    Ask your administrator to set up Apptainer on the node.

    !!! tip

        The machine used for the project, `p1068-rhel9-01-pool`, already has Apptainer installed.

To use the image on TSD

📥 Import the image to TSD

:   Go to [TSD's import page](https://data.tsd.usit.no/file-import/), and follow the instructions.
    Files will be available at `/tsd/<project-code>/data/durable/file-import/`.

:octicons-terminal-24: Run the image

:   Move the image to an appropriate location in TSD, and execute
    `apptainer run --no-home -B <data_manager_dir>:/mnt <image_name>` where `image_name` has the extension `.sif`.
    The `data_manager_dir` must be a directory containing either the Parquet cached or raw files for `decipher.data.DataManager`.

!!! info "Constructing the data"

    The data for the `data_manager_dir` may be simply the `lp_pid_fuzzy.csv` and `Folkereg_PID_fuzzy.csv` files.
    However, it is _highly_ recommended to cache the data using `decipher.data.DataManager`s Parquet format, as this gives much faster load times.

    Consult the `decipher` package documentation for details, but the easiest way to do this, is to run
    ```python
    from pathlib import Path
    from decipher.data import DataManager

    screening_data = Path(<screening_data>)
    dob_data = Path(<dob_data>)
    parquet_dir = Path(<parquet_dir>)

    # Read in from CSV
    data_manager = DataManager.read_from_csv(screening_data, dob_data, read_hpv=True) # (1)!

    # Store to Parquet
    data_manager.save_to_parquet(parquet_dir, engine="pyarrow")
    ```

    1.  :material-lightbulb: Note the `read_hpv=True`.

    which is taken from the [decipher documentation](https://github.com/Simula-Consulting/decipher_data_handler#usage).




[^1]: Formerly known as Singularity.

## :material-language-css3: CSS and :material-tailwind: Tailwind

We use [Tailwind](https://tailwindcss.com/docs/installation) for our CSS.
When making changes to the HTML templates, or if you need to add custom CSS, the generated CSS files must be regenerated.

Install Tailwind CLI with
```bash
npm install -D tailwindcss
```
and then run
```bash
npx tailwindcss -i apps/pilot/static/style_tailwind.css -o apps/pilot/static/style.css
npx tailwindcss -i apps/landing_page/static/style_tailwind.css -o apps/landing_page/static/style.css
```
to build the new CSS file.

!!! note

    Do not touch `apps/pilot/static/style.css` or `apps/landing_page/static/style.css`!
    These are auto-generated files.
    If you require custom css, add this to the appropriate `style_tailwind.css` file.
