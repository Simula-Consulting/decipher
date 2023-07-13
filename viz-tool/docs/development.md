# Development Guide

Welcome to the development guide. In the following sections, you will find detailed instructions on how to install and deploy the application, handle CSS with Tailwind, and add new filters to your application pages.

## Installation and Deployment :rocket:

!!! tip "Running on TSD"
    If you want to run the application on TSD, refer to the [Apptainer section below](#apptainer-formerly-singularity-support).

### ... with Poetry

Our project and its dependencies are managed with [Poetry](https://python-poetry.org/). Here's how to get it up and running:

1. Run `poetry install` to set up the environment.
2. Run `poetry run bokeh serve apps/pilot apps/landing_page` to launch the application.

### ... with Docker

If you prefer Docker, you can get started quickly with Docker Compose:
```bash
# Navigate to the viz-tool base folder
cd /path/to/viz-tool
# Start the application
docker compose up
```

### Apptainer (formerly Singularity) Support

To run our application on [TSD](https://www.uio.no/tjenester/it/forskning/sensitiv/), we use [Apptainer](https://apptainer.org/), formerly known as Singularity.

To convert the Docker image to Apptainer format, run the following command, replacing `viz-tool` with the name of your Docker image:

```bash
docker run \
-v /var/run/docker.sock:/var/run/docker.sock \
-v ~/Downloads:/output \
-v /tmp/build_singularity:/tmp \
--privileged -t --rm \
quay.io/singularity/docker2singularity \
viz-tool
```

1. Sets the output of the generated image to `~/Downloads`
2. May not be needed. In some cases the default cache volume runs out of space.

This command will create a `.sif` file in your `~/Downloads` folder.

??? question "Why use Docker to build Apptainer image"

    The above approach was chosen as it works well on any system and does not require Apptainer.
    If you have to Apptainer on your local x86 machine, feel free to build directly
    with Apptainer.

??? warning "Access to Apptainer"

    On TSD, Apptainer is not installed on the VMs by default.
    Ask your administrator to set up Apptainer on the node.

    !!! tip

        The machine used for the project, `p1068-rhel9-01-pool`, already has Apptainer installed.

To use the image on TSD:

**📥 Import the image to TSD**

:   Go to [TSD's import page](https://data.tsd.usit.no/file-import/), and follow the instructions to upload your `.sif` image.
    Files will be available at `/tsd/<project-code>/data/durable/file-import/`.

    :fire: tip:  Upload to your user's group, not the common group.

**:octicons-terminal-24: Run the image**

:   Move the image to an appropriate location in TSD, and execute
    `apptainer run --no-home -B <data_manager_dir>:/mnt <image_name>` where `image_name` has the extension `.sif`.
    The `data_manager_dir` must be a directory containing either the Parquet cached or raw files for `decipher.data.DataManager`.

!!! info "Constructing the data"

    The data for the `data_manager_dir` may simply be the `lp_pid_fuzzy.csv` and `Folkereg_PID_fuzzy.csv` files.
    However, it is _highly_ recommended to cache the data using `decipher.data.DataManager`s Parquet format, as this gives much faster load times.

    Consult the `decipher` package documentation for details, but the easiest way to do this, is to run
    ```python
    from pathlib import Path
    from decipher.data import DataManager

    # Set up data paths (2)
    screening_data = Path(<screening_data>)
    dob_data = Path(<dob_data>)
    parquet_dir = Path(<parquet_dir>)

    # Read in from CSV
    data_manager = DataManager.read_from_csv(screening_data, dob_data, read_hpv=True) # (1)!

    # Store to Parquet
    data_manager.save_to_parquet(parquet_dir, engine="pyarrow")
    ```

    1.  :material-lightbulb: Note the `read_hpv=True`.
    2.  Replace `<screening_data>`, `<dob_data>`, and `<parquet_dir>` with the actual paths to your input CSV files and the directory where you want to save the Parquet files.

    which is taken from the [decipher documentation](https://github.com/Simula-Consulting/decipher_data_handler#usage).


## Handling CSS and Tailwind

Our project uses [Tailwind](https://tailwindcss.com/docs/installation) for CSS. When you make changes to the HTML templates or need to add custom CSS, please remember to regenerate the CSS files.

To regenerate the CSS files:

1. Install the Tailwind CLI:
    ```bash
    npm install -D tailwindcss
    ```
2. Build the new CSS files:
    ```bash
    npx tailwindcss -i apps/pilot/static/style_tailwind.css -o apps/pilot/static/style.css
    npx tailwindcss -i apps/landing_page/static/style_tailwind.css -o apps/landing_page/static/style.css
    ```

!!! note "Note on custom CSS"
    Please do not modify the `apps/pilot/static/style.css` or `apps/landing_page/static/style.css` files directly. These are auto-generated files. If you need to add custom CSS, add it to the appropriate `style_tailwind.css` file.

## Adding New Filters

### On the Landing Page

Each filter on the landing page is custom-built. To add a new filter, you need to extend the code in `apps.landing_page.main.py::LandingPageFilter`.
For filters that are simply on or off (as opposed to having a value or range of values), the easiest is to modify the checkboxes.

### On the Main Page

A variety of filter types are defined in `viz_tool.backend`, all inheriting from `viz_tool.backend.BaseFilter`. To utilize these filters, add them to the app's `source_manager`. This is done in the `apps.pilot.main.py::_get_filters` method.
