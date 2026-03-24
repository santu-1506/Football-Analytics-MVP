# Entry point — reads video, runs full pipeline, saves annotated output

import numpy as np
from utils import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner


def main():
    # ── 1. Read video ──────────────────────────────────────────────────────────
    video_frames = read_video('input_videos/sample.mp4')

    # ── 2. Track objects ───────────────────────────────────────────────────────
    tracker = Tracker('yolo11l.pt')
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
    )

    # Add foot/center position to every tracked object
    tracker.add_position_to_tracks(tracks)

    # ── 3. Interpolate missing ball positions ──────────────────────────────────
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # ── 4. Assign teams ────────────────────────────────────────────────────────
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = \
                team_assigner.team_colors[team]

    # ── 5. Assign ball possession ──────────────────────────────────────────────
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(
                tracks['players'][frame_num][assigned_player]['team']
            )
        else:
            # No one has ball — carry forward last known possession
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)

    # ── 6. Draw annotations ────────────────────────────────────────────────────
    output_video_frames = tracker.draw_annotations(
        video_frames,
        tracks,
        team_ball_control
    )

    # ── 7. Save output ─────────────────────────────────────────────────────────
    save_video(output_video_frames, 'output_videos/output.mp4')
    print("✅ Done! Output saved to output_videos/output.mp4")


if __name__ == '__main__':
    main()