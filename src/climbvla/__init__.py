def main() -> None:
    """The `climbvla` command (uv run climbvla climb.mp4): see climbvla.analyze."""
    # imported here, so that importing the package stays light
    from climbvla.analyze import main as analyze_main

    analyze_main()
