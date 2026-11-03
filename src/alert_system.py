"""
Intelligent Audio and Visual Alert System for Helmet Detection.
Provides rate-limited warning sounds, configurable cooldowns, and non-blocking playback.
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AlertSoundSystem:
    """
    Dispatches audio alerts with cooldown protection and background threading.
    Supports custom WAV files with automatic fallback to system beeps.
    """

    def __init__(
        self,
        sound_enabled: bool = True,
        cooldown_seconds: float = 5.0,
        sound_file: Optional[Path] = None
    ):
        self.sound_enabled = sound_enabled
        self.cooldown_seconds = cooldown_seconds
        self.sound_file = Path(sound_file) if sound_file else (PROJECT_ROOT / "alerts" / "alarm.wav")
        self.last_alert_time = 0.0
        self._lock = threading.Lock()

    def toggle_sound(self, enable: Optional[bool] = None) -> bool:
        """Toggles or sets audio alert state."""
        with self._lock:
            if enable is None:
                self.sound_enabled = not self.sound_enabled
            else:
                self.sound_enabled = enable
            print(f"[Alert System] Audio alerts {'ENABLED' if self.sound_enabled else 'DISABLED'}.")
            return self.sound_enabled

    def set_cooldown(self, seconds: float):
        """Sets the minimum interval between consecutive sound alerts."""
        with self._lock:
            self.cooldown_seconds = max(1.0, float(seconds))

    def trigger_audio_alert(self, force: bool = False) -> bool:
        """
        Triggers an audio alert if sound is enabled and cooldown interval has passed.
        Returns True if an alert sound was played, False otherwise.
        """
        if not self.sound_enabled:
            return False

        current_time = time.time()
        with self._lock:
            if not force and (current_time - self.last_alert_time) < self.cooldown_seconds:
                return False  # Cooldown active
            self.last_alert_time = current_time

        # Play in non-blocking thread so video feed remains responsive
        threading.Thread(target=self._play_sound_worker, daemon=True).start()
        return True

    def _play_sound_worker(self):
        """Background worker executing platform audio."""
        try:
            # 1. Try Windows WAV playback if custom audio file exists
            if self.sound_file.exists() and sys.platform == "win32":
                import winsound
                winsound.PlaySound(str(self.sound_file), winsound.SND_FILENAME | winsound.SND_NODEFAULT)
                return

            # 2. Graceful Fallback: Built-in Windows system beep (1200 Hz for 350 ms)
            if sys.platform == "win32":
                import winsound
                winsound.Beep(1200, 350)
                return

            # 3. Cross-platform terminal bell fallback
            print("\a", end="", flush=True)

        except Exception as e:
            # Missing audio drivers or headless environment - handle gracefully
            print(f"[Alert System] Audio playback fallback (non-fatal): {e}")


if __name__ == "__main__":
    print("Testing AlertSoundSystem...")
    alerts = AlertSoundSystem(cooldown_seconds=2.0)
    print("Trigger 1 (should play):", alerts.trigger_audio_alert())
    time.sleep(0.5)
    print("Trigger 2 (should be blocked by cooldown):", alerts.trigger_audio_alert())
    time.sleep(2.0)
    print("Trigger 3 (should play after cooldown):", alerts.trigger_audio_alert())
    time.sleep(0.5)

