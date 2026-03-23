import sys

with open("cli.py", "r") as f:
    content = f.read()

# 1. Update FocusZone enum
old_enum = """class FocusZone(Enum):
    MENU = auto()
    COLUMNS = auto()
    PLAYBACK = auto()"""

new_enum = """class FocusZone(Enum):
    MENU = auto()
    POPULATION = auto()
    BANDS = auto()
    PLAYBACK = auto()"""
content = content.replace(old_enum, new_enum)

# 2. Update __init__
old_init = "self.focus_zone = FocusZone.COLUMNS"
new_init = "self.focus_zone = FocusZone.POPULATION"
content = content.replace(old_init, new_init)

# 3. Replace navigate and handlers
old_nav_start = "    def navigate(self, key: str):"
old_nav_end = """        elif key == KEY_RIGHT:
            if self.sample_rate and self.processed_audio is not None:
                self.play_idx = min(self.processed_audio.shape[1] - 1, self.play_idx + self.sample_rate)"""

new_nav = """    def navigate(self, key: str):
        if self.editing:
            if key == KEY_ENTER:
                try: self.set_value(float(self.edit_buffer))
                except: pass
                self.editing = False; self.edit_buffer = ""
            elif key == KEY_ESC:
                self.editing = False; self.edit_buffer = ""
            elif key == KEY_BACKSPACE:
                self.edit_buffer = self.edit_buffer[:-1]
            elif key and len(key) == 1 and key != " ":
                self.edit_buffer += key
            return

        if key == KEY_TAB and self.mode == "EVOLUTION":
            zones = list(FocusZone)
            idx = zones.index(self.focus_zone)
            self.focus_zone = zones[(idx + 1) % len(zones)]
            return

        if key == KEY_SPACE:
            self.toggle_playback()
            return

        # If in FILE_PICKER, handle its inputs directly
        if self.mode == "FILE_PICKER":
            if key == KEY_UP:
                self.selected_file_idx = (self.selected_file_idx - 1) % max(1, len(self.file_list))
            elif key == KEY_DOWN:
                self.selected_file_idx = (self.selected_file_idx + 1) % max(1, len(self.file_list))
            elif key == KEY_ENTER:
                if self.file_list:
                    item = self.file_list[self.selected_file_idx]
                    if item["type"] == "dir":
                        self.current_path = os.path.abspath(os.path.join(self.current_path, item["name"]))
                        self.refresh_file_list()
                    else:
                        self.load_audio(item["name"])
            elif key == KEY_BACKSPACE:
                self.current_path = os.path.abspath(os.path.join(self.current_path, ".."))
                self.refresh_file_list()
            return

        # Evolution Global Shortcuts
        if key == KEY_L:
            self.mode = "FILE_PICKER"
            self.refresh_file_list()
            return
        elif key == KEY_BRACKET_LEFT:
            self.selected_band_idx = (self.selected_band_idx - 1) % len(self.population.mixes[0].bands)
            self.selected_module_idx = 0; self.selected_param_idx = 0; self.editing_crossover = False
        elif key == KEY_BRACKET_RIGHT:
            self.selected_band_idx = (self.selected_band_idx + 1) % len(self.population.mixes[0].bands)
            self.selected_module_idx = 0; self.selected_param_idx = 0; self.editing_crossover = False

        if self.focus_zone == FocusZone.MENU:
            self._handle_menu_input(key)
        elif self.focus_zone == FocusZone.POPULATION:
            self._handle_population_input(key)
        elif self.focus_zone == FocusZone.BANDS:
            self._handle_bands_input(key)
        elif self.focus_zone == FocusZone.PLAYBACK:
            self._handle_playback_input(key)

    def _handle_menu_input(self, key: str):
        if not self.header_menu.items: return
        if key == KEY_LEFT:
            self.header_menu.selected_index = (self.header_menu.selected_index - 1) % len(self.header_menu.items)
        elif key == KEY_RIGHT:
            self.header_menu.selected_index = (self.header_menu.selected_index + 1) % len(self.header_menu.items)
        elif key == KEY_ENTER:
            item = self.header_menu.items[self.header_menu.selected_index]
            if item.action:
                item.action()

    def _handle_playback_input(self, key: str):
        if key == KEY_LEFT:
            if self.sample_rate:
                self.play_idx = max(0, self.play_idx - self.sample_rate)
        elif key == KEY_RIGHT:
            if self.sample_rate and self.processed_audio is not None:
                self.play_idx = min(self.processed_audio.shape[1] - 1, self.play_idx + self.sample_rate)"""

content = content[:content.find(old_nav_start)] + new_nav + content[content.find(old_nav_end) + len(old_nav_end):]

# 4. Replace _handle_column_input with _handle_population_input and _handle_bands_input
old_col_start = "    def _handle_column_input(self, key: str):"
old_col_end = """                p = self.get_selected_param()
                if p:
                    self.editing = True
                    self.edit_buffer = str(round(p.current_value, 2))"""

new_col = """    def _handle_population_input(self, key: str):
        if key == KEY_LEFT:
            self.selected_mix_idx = (self.selected_mix_idx - 1) % len(self.population.mixes)
            if self.is_playing: self.request_reprocessing()
        elif key == KEY_RIGHT:
            self.selected_mix_idx = (self.selected_mix_idx + 1) % len(self.population.mixes)
            if self.is_playing: self.request_reprocessing()
        elif key == KEY_ENTER:
            self.population.generate_next_generation(self.selected_mix_idx, 0.5, 0.5)
            if self.is_playing: self.request_reprocessing()

    def _handle_bands_input(self, key: str):
        mix = self.population.mixes[self.selected_mix_idx % len(self.population.mixes)]
        band = mix.bands[self.selected_band_idx]
        if key == KEY_UP:
            if self.selected_param_idx > 0:
                self.selected_param_idx -= 1
            elif self.selected_module_idx > 0:
                self.selected_module_idx -= 1
                self.selected_param_idx = len(band.modules[self.selected_module_idx].parameters) - 1
            elif not self.editing_crossover:
                self.editing_crossover = True
        elif key == KEY_DOWN:
            if self.editing_crossover:
                self.editing_crossover = False
                self.selected_module_idx = 0
                self.selected_param_idx = 0
            elif band.modules:
                mod = band.modules[self.selected_module_idx]
                if self.selected_param_idx < len(mod.parameters) - 1:
                    self.selected_param_idx += 1
                elif self.selected_module_idx < len(band.modules) - 1:
                    self.selected_module_idx += 1
                    self.selected_param_idx = 0
        elif key == KEY_LEFT:
            self.adjust_value(-1)
        elif key == KEY_RIGHT:
            self.adjust_value(1)
        elif key == KEY_ENTER:
            p = self.get_selected_param()
            if p:
                self.editing = True
                self.edit_buffer = str(round(p.current_value, 2))"""
content = content[:content.find(old_col_start)] + new_col + content[content.find(old_col_end) + len(old_col_end):]

# 5. Fix render highlights
content = content.replace("self.focus_zone == FocusZone.COLUMNS and self.focus_row == 0", "self.focus_zone == FocusZone.POPULATION")
content = content.replace("self.focus_zone == FocusZone.COLUMNS and self.focus_row == 1", "self.focus_zone == FocusZone.BANDS")
content = content.replace("self.focus_zone == FocusZone.COLUMNS and is_band_sel", "self.focus_zone == FocusZone.BANDS and is_band_sel")

with open("cli.py", "w") as f:
    f.write(content)
