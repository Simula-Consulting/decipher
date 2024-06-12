# Reference

**Organization of the code**

The code for the app is mainly organized into [`viz_tool.backend`][viz_tool.backend] and [`viz_tool.frontend`][viz_tool.frontend].
The former sets up the data structure, while the latter offers various components, like the Lexis plot and histograms.

From the backend, the [`SourceManager`][viz_tool.backend.SourceManager] is the most convenient object, and it is used by most front end components.

::: viz_tool.backend
    options:
      members:
        - SourceManager
      show_root_heading: true
      docstring_options:
        ignore_init_summary: true
      merge_init_into_class: true

::: viz_tool.frontend
    options:
      members:
        - HistogramPlot
        - LabelSelectedMixin
        - LexisPlot
      show_root_heading: true

::: viz_tool.settings
    options:
      show_if_no_docstring: true
      show_root_heading: true
