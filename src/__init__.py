from chimerax.core.toolshed import BundleAPI


class _AlphaLensAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd
        cmd.register_commands(logger)

    @staticmethod
    def start_tool(session, bi, ti):
        from .cmd import start_tool_from_menu
        return start_tool_from_menu(session)


bundle_api = _AlphaLensAPI()
