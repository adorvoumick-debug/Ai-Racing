import math
import os
import pygame

from game.track import Track
from game.car import Car
from learning.q_learning import QLearningAgent


pygame.init()

WIDTH = 1000
HEIGHT = 700
FPS = 60

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI RACING | Adaptive Racing Lab")

clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 20, bold=True)
medium_font = pygame.font.SysFont("Arial", 35, bold=True)
big_font = pygame.font.SysFont("Arial", 60, bold=True)
small_font = pygame.font.SysFont("Arial", 16)

TOTAL_LAPS = 3
OFFROAD_LIMIT = 20

AI_MAX_SPEED = 2.8
AI_MIN_CORNER_SPEED = 0.9
AI_ACCELERATION = 0.10

track = Track(screen, 1)

agent = QLearningAgent()


def set_ai_memory_file(track_number):
    project_folder = os.path.dirname(os.path.abspath(__file__))

    agent.memory_file = os.path.join(
        project_folder,
        f"ai_memory_track_{track_number}.json"
    )

    agent.q_table = {}
    agent.epsilon = 0.08
    agent.load_q_table()


def wrap_angle(angle):
    return (angle + 180) % 360 - 180


def heading_error(car):
    desired = track.get_desired_heading(car.x, car.y)
    return wrap_angle(desired - car.angle)


def make_state(car):
    sensors = track.get_sensor_readings(
        car.x,
        car.y,
        car.angle
    )

    error = heading_error(car)

    state = agent.get_state(
        sensors,
        car.speed,
        error
    )

    return state, sensors, error


def create_player():
    x, y, angle = track.get_start_position()

    car = Car(x, y)
    car.angle = angle
    car.color = (40, 130, 255)

    return car


def create_ai():
    x, y, angle = track.get_start_position()

    car = Car(x, y + 18)
    car.angle = angle
    car.color = (230, 50, 50)

    return car


def get_corner_severity(error, sensors):
    severity = abs(error)

    front = sensors[len(sensors) // 2]

    if front < 25:
        severity += 25
    elif front < 45:
        severity += 12

    return severity


def control_ai(car):
    state, sensors, error = make_state(car)

    action = agent.choose_action(state)

    # Strong heading correction.
    if error > 15:
        action = 2
    elif error < -15:
        action = 1
    elif error > 6:
        action = 2
    elif error < -6:
        action = 1

    severity = get_corner_severity(error, sensors)

    if severity < 8:
        car.speed += AI_ACCELERATION
    elif severity < 16:
        car.speed += 0.12
    elif severity < 25:
        car.speed += 0.04
    else:
        car.speed *= 0.86
        car.speed = max(
            car.speed,
            AI_MIN_CORNER_SPEED
        )

    front = sensors[len(sensors) // 2]

    if front < 25:
        car.speed *= 0.72
    elif front < 45:
        car.speed *= 0.88

    if action == 1:
        car.angle -= car.rotation_speed
    elif action == 2:
        car.angle += car.rotation_speed

    car.speed = max(
        0.0,
        min(AI_MAX_SPEED, car.speed)
    )

    car.update()

    return state, sensors, error, severity, action


def recover_ai(car):
    if track.is_on_road(car.x, car.y):
        return

    desired = track.get_desired_heading(
        car.x,
        car.y
    )

    error = wrap_angle(
        desired - car.angle
    )

    if error > 0:
        car.angle += car.rotation_speed * 1.5
    else:
        car.angle -= car.rotation_speed * 1.5

    car.speed *= 0.70

    progress, distance = track.get_nearest_progress(
        car.x,
        car.y
    )

    if distance > track.road_width * 0.95:
        index = int(
            progress * len(track.samples)
        ) % len(track.samples)

        target_x, target_y = track.samples[index]

        car.x += (target_x - car.x) * 0.08
        car.y += (target_y - car.y) * 0.08


def calculate_progress_change(previous, current):
    change = current - previous

    if change > 0.5:
        change -= 1.0
    elif change < -0.5:
        change += 1.0

    return change


def update_progress(car, previous_progress):
    current_progress, distance = track.get_nearest_progress(
        car.x,
        car.y
    )

    change = calculate_progress_change(
        previous_progress,
        current_progress
    )

    return current_progress, change


def crossed_finish_line(
    previous_progress,
    current_progress,
    has_left_start
):
    """
    A lap is counted only after:
      1. The car has left the start/finish area.
      2. It travels almost the entire centerline.
      3. It crosses from the end of the track back to the beginning.

    Therefore simply driving over the line near the beginning of the
    race cannot create a lap.
    """
    if not has_left_start:
        return False

    return track.is_past_start_line_forward(
        previous_progress,
        current_progress
    )


def reset_race():
    global player
    global ai_car
    global player_lap
    global ai_lap
    global player_progress
    global ai_progress
    global player_has_left_start
    global ai_has_left_start
    global ai_offroad_frames
    global ai_crashes
    global race_finished
    global winner
    global race_started
    global countdown_start
    global race_start_time

    player = create_player()
    ai_car = create_ai()

    player_lap = 0
    ai_lap = 0

    player_progress = track.get_nearest_progress(
        player.x,
        player.y
    )[0]

    ai_progress = track.get_nearest_progress(
        ai_car.x,
        ai_car.y
    )[0]

    # These become True only after the car has travelled away from
    # the starting section of the circuit.
    player_has_left_start = False
    ai_has_left_start = False

    ai_offroad_frames = 0
    ai_crashes = 0

    race_finished = False
    winner = ""

    race_started = False
    countdown_start = pygame.time.get_ticks()
    race_start_time = 0


def select_track():
    global track

    selected = 1

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_1, pygame.K_KP1):
                    selected = 1
                elif event.key in (pygame.K_2, pygame.K_KP2):
                    selected = 2
                elif event.key in (pygame.K_3, pygame.K_KP3):
                    selected = 3
                elif event.key in (
                    pygame.K_RETURN,
                    pygame.K_SPACE
                ):
                    track = Track(screen, selected)
                    set_ai_memory_file(selected)
                    return
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise SystemExit

        screen.fill((18, 22, 28))

        title = big_font.render(
            "AI RACING",
            True,
            (255, 255, 255)
        )

        screen.blit(
            title,
            (
                WIDTH // 2 - title.get_width() // 2,
                60
            )
        )

        subtitle = medium_font.render(
            "SELECT YOUR TRACK",
            True,
            (80, 210, 255)
        )

        screen.blit(
            subtitle,
            (
                WIDTH // 2 - subtitle.get_width() // 2,
                135
            )
        )

        tracks = Track.available_tracks()

        for number, data in tracks.items():
            y = 225 + (number - 1) * 105

            selected_box = number == selected

            box_color = (
                (55, 85, 105)
                if selected_box
                else (35, 40, 48)
            )

            border_color = (
                (80, 220, 255)
                if selected_box
                else (100, 100, 100)
            )

            box = pygame.Rect(
                210,
                y,
                580,
                82
            )

            pygame.draw.rect(
                screen,
                box_color,
                box,
                border_radius=12
            )

            pygame.draw.rect(
                screen,
                border_color,
                box,
                3,
                border_radius=12
            )

            number_text = medium_font.render(
                str(number),
                True,
                (255, 220, 50)
            )

            screen.blit(
                number_text,
                (235, y + 20)
            )

            name = font.render(
                data["name"],
                True,
                (255, 255, 255)
            )

            screen.blit(
                name,
                (300, y + 13)
            )

            desc = small_font.render(
                data["description"],
                True,
                (190, 200, 210)
            )

            screen.blit(
                desc,
                (300, y + 47)
            )

        help_text = font.render(
            "Press 1-3 to select   |   ENTER / SPACE to start",
            True,
            (220, 220, 220)
        )

        screen.blit(
            help_text,
            (
                WIDTH // 2 - help_text.get_width() // 2,
                570
            )
        )

        pygame.display.flip()


def draw_text(
    text,
    x,
    y,
    color=(255, 255, 255),
    selected_font=None
):
    if selected_font is None:
        selected_font = font

    image = selected_font.render(
        text,
        True,
        color
    )

    screen.blit(image, (x, y))


def draw_panel(rect, alpha=205, border=(75, 205, 245)):
    panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(panel, (13, 19, 28, alpha), panel.get_rect(), border_radius=12)
    pygame.draw.rect(panel, (*border, 235), panel.get_rect(), 2, border_radius=12)
    screen.blit(panel, rect.topleft)


def draw_bar(x, y, width, height, value, maximum, color):
    pygame.draw.rect(screen, (37, 45, 54), (x, y, width, height), border_radius=height // 2)
    fill = int(width * max(0.0, min(1.0, value / maximum))) if maximum else 0
    if fill:
        pygame.draw.rect(screen, color, (x, y, fill, height), border_radius=height // 2)


def draw_hud():
    # Race information panel
    left = pygame.Rect(14, 14, 250, 176)
    draw_panel(left)
    draw_text("RACE CONTROL", 28, 24, (78, 220, 255), small_font)
    pygame.draw.line(screen, (67, 82, 96), (28, 48), (248, 48), 1)
    draw_text(track.track_data['name'], 28, 57, (245, 248, 255), font)
    draw_text(f"PLAYER  {player_lap}/{TOTAL_LAPS} LAPS", 28, 87, (100, 180, 255), small_font)
    draw_bar(28, 108, 215, 7, player_lap, TOTAL_LAPS, (60, 150, 255))
    draw_text(f"AI       {ai_lap}/{TOTAL_LAPS} LAPS", 28, 124, (255, 110, 105), small_font)
    draw_bar(28, 145, 215, 7, ai_lap, TOTAL_LAPS, (245, 75, 75))
    draw_text(f"AI crashes: {ai_crashes}", 28, 160, (205, 215, 225), small_font)

    # AI learning telemetry panel
    right = pygame.Rect(WIDTH - 244, 14, 230, 130)
    draw_panel(right, border=(255, 105, 95))
    draw_text("AI TELEMETRY", right.x + 14, 24, (255, 130, 120), small_font)
    draw_text(f"Learned states  {len(agent.q_table)}", right.x + 14, 52, (245, 248, 255), small_font)
    draw_text(f"Exploration     {agent.epsilon:.3f}", right.x + 14, 77, (210, 220, 230), small_font)
    draw_text("SENSORS: ACTIVE", right.x + 14, 103, (90, 230, 190), small_font)

    # Speedometer panel
    speed_box = pygame.Rect(WIDTH - 218, HEIGHT - 132, 204, 116)
    draw_panel(speed_box, alpha=220, border=(80, 180, 255))
    draw_text("PLAYER SPEED", speed_box.x + 14, speed_box.y + 10, (160, 205, 255), small_font)
    speed_value = abs(player.speed)
    draw_text(f"{speed_value * 22:03.0f}", speed_box.x + 15, speed_box.y + 29,
              (255, 255, 255), big_font)
    draw_text("KM/H", speed_box.x + 119, speed_box.y + 56, (170, 190, 210), small_font)
    draw_bar(speed_box.x + 15, speed_box.y + 91, 174, 8, speed_value, player.max_speed,
             (80, 220, 255) if speed_value < 3.5 else (255, 180, 65))

    # Race timer and controls footer
    if race_started and not race_finished:
        elapsed = (pygame.time.get_ticks() - race_start_time) / 1000
        timer = f"TIME  {int(elapsed // 60):02d}:{elapsed % 60:05.2f}"
        timer_img = medium_font.render(timer, True, (255, 255, 255))
        timer_bg = pygame.Rect(WIDTH//2 - timer_img.get_width()//2 - 15, 14,
                               timer_img.get_width()+30, 48)
        draw_panel(timer_bg, alpha=190, border=(255, 210, 90))
        screen.blit(timer_img, (timer_bg.x + 15, timer_bg.y + 4))

    hint = small_font.render("ARROWS: DRIVE     R: RESTART AFTER FINISH     M: TRACK MENU", True, (235, 240, 245))
    hint_bg = pygame.Surface((hint.get_width()+22, hint.get_height()+12), pygame.SRCALPHA)
    pygame.draw.rect(hint_bg, (10, 15, 22, 170), hint_bg.get_rect(), border_radius=8)
    hint_bg.blit(hint, (11, 6))
    screen.blit(hint_bg, (WIDTH//2-hint_bg.get_width()//2, HEIGHT-hint_bg.get_height()-10))


def draw_countdown():
    elapsed = (pygame.time.get_ticks() - countdown_start) / 1000
    if elapsed < 1:
        message = "3"
    elif elapsed < 2:
        message = "2"
    elif elapsed < 3:
        message = "1"
    elif elapsed < 4:
        message = "GO!"
    else:
        return True

    shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    shade.fill((4, 8, 14, 95))
    screen.blit(shade, (0, 0))
    label = medium_font.render("GET READY" if message != "GO!" else "RACE STARTED", True, (210, 225, 240))
    number_color = (100, 225, 255) if message != "GO!" else (130, 255, 160)
    number = big_font.render(message, True, number_color)
    panel = pygame.Rect(WIDTH//2-150, HEIGHT//2-112, 300, 224)
    draw_panel(panel, alpha=225, border=number_color)
    screen.blit(label, (WIDTH//2-label.get_width()//2, panel.y+28))
    screen.blit(number, (WIDTH//2-number.get_width()//2, panel.y+78))
    return False


# ---------------- START ----------------

set_ai_memory_file(1)
select_track()
reset_race()

running = True
paused = False
pause_started_at = 0

while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            # P toggles pause/resume. Shift the countdown clock when resuming
            # so the countdown does not continue invisibly while paused.
            if event.key == pygame.K_p and not race_finished:
                if paused:
                    paused_duration = pygame.time.get_ticks() - pause_started_at
                    countdown_start += paused_duration
                    paused = False
                else:
                    paused = True
                    pause_started_at = pygame.time.get_ticks()

            # R resets the current race at any time without changing the
            # selected track or deleting saved AI learning memory.
            if event.key == pygame.K_r:
                reset_race()
                paused = False

            if event.key == pygame.K_m and race_finished:
                select_track()
                reset_race()

            if event.key == pygame.K_ESCAPE and race_finished:
                select_track()
                reset_race()

    if paused:
        pause_overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pause_overlay.fill((0, 0, 0, 155))
        screen.blit(pause_overlay, (0, 0))
        pause_title = big_font.render("PAUSED", True, (255, 255, 255))
        pause_hint = font.render("P = Resume    R = Restart race", True, (110, 220, 255))
        screen.blit(pause_title, pause_title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 30)))
        screen.blit(pause_hint, pause_hint.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 35)))
        pygame.display.flip()
        continue

    if not race_started:
        if draw_countdown():
            race_started = True
            race_start_time = pygame.time.get_ticks()

    elif not race_finished:

        # ---------------- PLAYER ----------------

        player.handle_input()
        player.update()

        if not track.is_on_road(
            player.x,
            player.y
        ):
            player = create_player()

            player_progress = track.get_nearest_progress(
                player.x,
                player.y
            )[0]

            # Reset start-line state after a crash/off-road reset.
            player_has_left_start = False

        old_player_progress = player_progress

        (
            player_progress,
            player_change
        ) = update_progress(
            player,
            player_progress
        )

        # Arm lap counting only after the player has moved away from
        # the actual start/finish line, regardless of track progress value.
        player_relative_progress = track.get_progress_from_start(player_progress)
        if 0.10 < player_relative_progress < 0.90:
            player_has_left_start = True

        if crossed_finish_line(
            old_player_progress,
            player_progress,
            player_has_left_start
        ):
            player_lap += 1

        # ---------------- AI ----------------

        old_ai_progress = ai_progress

        (
            state,
            sensors,
            error,
            severity,
            action
        ) = control_ai(ai_car)

        recover_ai(ai_car)

        (
            ai_progress,
            ai_change
        ) = update_progress(
            ai_car,
            ai_progress
        )

        ai_relative_progress = track.get_progress_from_start(ai_progress)
        if 0.10 < ai_relative_progress < 0.90:
            ai_has_left_start = True

        if crossed_finish_line(
            old_ai_progress,
            ai_progress,
            ai_has_left_start
        ):
            ai_lap += 1

        # ---------------- AI LEARNING ----------------

        reward = ai_change * 120.0

        reward += max(
            0.0,
            1.0 - abs(error) / 45.0
        ) * 0.20

        if track.is_on_road(
            ai_car.x,
            ai_car.y
        ):
            ai_offroad_frames = 0
            reward += 0.30
        else:
            ai_offroad_frames += 1
            reward -= 3.0

        if severity > 45:
            reward -= 0.75

        if severity > 25 and ai_car.speed <= 3.5:
            reward += 0.15

        crashed = ai_offroad_frames >= OFFROAD_LIMIT

        if crashed:
            reward -= 25.0

        next_state, _, _ = make_state(ai_car)

        # IMPORTANT:
        # Learn from the exact action that was actually used.
        agent.learn(
            state,
            action,
            reward,
            next_state,
            done=crashed
        )

        agent.reduce_exploration()

        if crashed:
            ai_crashes += 1
            ai_car = create_ai()

            ai_progress = track.get_nearest_progress(
                ai_car.x,
                ai_car.y
            )[0]

            ai_has_left_start = False
            ai_offroad_frames = 0

        # ---------------- WINNER ----------------

        if player_lap >= TOTAL_LAPS:
            race_finished = True
            winner = "PLAYER WINS!"

        elif ai_lap >= TOTAL_LAPS:
            race_finished = True
            winner = "AI WINS!"

    # ---------------- DRAW ----------------

    track.draw()

    sensors = track.get_sensor_readings(
        ai_car.x,
        ai_car.y,
        ai_car.angle
    )

    track.draw_sensors(
        ai_car.x,
        ai_car.y,
        ai_car.angle,
        sensors
    )

    player.draw(screen)
    ai_car.draw(screen)

    draw_hud()

    if race_finished:
        overlay = pygame.Surface(
            (WIDTH, HEIGHT),
            pygame.SRCALPHA
        )

        overlay.fill((0, 0, 0, 165))
        screen.blit(overlay, (0, 0))

        draw_text(
            winner,
            WIDTH // 2 - 210,
            HEIGHT // 2 - 90,
            selected_font=big_font
        )

        draw_text(
            "R = race again",
            WIDTH // 2 - 110,
            HEIGHT // 2 + 10,
            selected_font=medium_font
        )

        draw_text(
            "M / ESC = track selection",
            WIDTH // 2 - 150,
            HEIGHT // 2 + 60,
            selected_font=font
        )

    pygame.display.flip()

agent.save_q_table()
pygame.quit()
