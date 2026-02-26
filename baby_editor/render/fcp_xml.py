"""Generate Final Cut Pro XML (version 5) compatible with DaVinci Resolve and Premiere Pro."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom


class FCPXMLGenerator:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.timebase = int(fps)

    def generate(self, edit_plan: dict, output_path: str) -> str:
        """Generate FCP XML from edit_plan dict.

        Returns the output path.
        """
        root = ET.Element("xmeml", version="5")
        sequence = ET.SubElement(root, "sequence")

        name = ET.SubElement(sequence, "name")
        name.text = edit_plan.get("parameters", {}).get("date", "Baby Edit")

        rate = ET.SubElement(sequence, "rate")
        ET.SubElement(rate, "timebase").text = str(self.timebase)
        ET.SubElement(rate, "ntsc").text = "FALSE"

        actual_dur = edit_plan.get("parameters", {}).get("actual_duration", 60)
        duration = ET.SubElement(sequence, "duration")
        duration.text = str(int(actual_dur * self.fps))

        media = ET.SubElement(sequence, "media")
        video_track = ET.SubElement(ET.SubElement(media, "video"), "track")
        audio_track = ET.SubElement(ET.SubElement(media, "audio"), "track")

        timeline_pos = 0

        for clip in edit_plan.get("timeline", []):
            in_point = clip.get("in_point", clip.get("in", clip.get("start", 0)))
            out_point = clip.get("out_point", clip.get("out", clip.get("end", 0)))
            source_video = clip.get("source_video", clip.get("video", "unknown"))
            source_path = clip.get("source_path", "")

            in_frames = int(in_point * self.fps)
            out_frames = int(out_point * self.fps)
            dur_frames = out_frames - in_frames

            # Video clip item
            clipitem = ET.SubElement(video_track, "clipitem")
            ET.SubElement(clipitem, "name").text = source_video
            ET.SubElement(clipitem, "duration").text = str(dur_frames)

            clip_rate = ET.SubElement(clipitem, "rate")
            ET.SubElement(clip_rate, "timebase").text = str(self.timebase)

            ET.SubElement(clipitem, "start").text = str(timeline_pos)
            ET.SubElement(clipitem, "end").text = str(timeline_pos + dur_frames)
            ET.SubElement(clipitem, "in").text = str(in_frames)
            ET.SubElement(clipitem, "out").text = str(out_frames)

            # File reference
            file_elem = ET.SubElement(clipitem, "file")
            file_id = source_video.replace(".", "_")
            file_elem.set("id", file_id)
            ET.SubElement(file_elem, "name").text = source_video
            if source_path:
                ET.SubElement(file_elem, "pathurl").text = (
                    f"file://localhost{source_path}"
                )

            # Transition
            trans = clip.get("transition_in", {})
            trans_type = trans.get("type")
            trans_dur = trans.get("duration", 0)
            if trans_type in ("dissolve", "fade_black") and trans_dur > 0:
                transition = ET.SubElement(video_track, "transition")
                half_frames = int(trans_dur * self.fps / 2)
                ET.SubElement(transition, "alignment").text = "center"
                ET.SubElement(transition, "start").text = str(
                    max(0, timeline_pos - half_frames)
                )
                ET.SubElement(transition, "end").text = str(
                    timeline_pos + half_frames
                )

                effect = ET.SubElement(transition, "effect")
                if trans_type == "dissolve":
                    ET.SubElement(effect, "name").text = "Cross Dissolve"
                    ET.SubElement(effect, "effectid").text = "Cross Dissolve"
                else:
                    ET.SubElement(effect, "name").text = "Dip to Black"
                    ET.SubElement(effect, "effectid").text = "Dip to Color"

            # Audio clip item (mirrors video)
            audio_clip = ET.SubElement(audio_track, "clipitem")
            ET.SubElement(audio_clip, "name").text = source_video
            ET.SubElement(audio_clip, "start").text = str(timeline_pos)
            ET.SubElement(audio_clip, "end").text = str(timeline_pos + dur_frames)
            ET.SubElement(audio_clip, "in").text = str(in_frames)
            ET.SubElement(audio_clip, "out").text = str(out_frames)

            audio_file = ET.SubElement(audio_clip, "file")
            audio_file.set("id", file_id)

            timeline_pos += dur_frames

        # Pretty print and write
        xml_str = minidom.parseString(
            ET.tostring(root, encoding="unicode")
        ).toprettyxml(indent="  ")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_str)

        return output_path
