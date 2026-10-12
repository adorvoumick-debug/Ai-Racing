import math
import random
import pygame


class Track:
    """
    Multi-track racing environment.

    The start/finish line is always generated from the centerline:
    - its center is exactly on the middle of the road
    - it is perpendicular to the road direction
    - lap counting can therefore use the centerline progress consistently
    """

    WIDTH = 1000
    HEIGHT = 700

    TRACKS = {
        1: {
            "name": "Beginner Circuit",
            "description": "Wide and smooth",
            "width": 92,
            "points": [
                (180, 180), (300, 125), (520, 125), (760, 170),
                (835, 285), (790, 455), (620, 535), (390, 540),
                (210, 465), (150, 330)
            ],
        },
        2: {
            "name": "Technical Circuit",
            "description": "More corners",
            "width": 82,
            "points": [
                (170, 190), (300, 130), (470, 155), (585, 115),
                (790, 180), (820, 315), (720, 360), (800, 485),
                (610, 550), (470, 485), (330, 550), (170, 450),
                (220, 330), (135, 270)
            ],
        },
        3: {
            "name": "Challenge Circuit",
            "description": "Tight and difficult",
            "width": 70,
            "points": [
                (160, 185), (275, 120), (390, 170), (500, 105),
                (650, 145), (810, 230), (730, 315), (845, 405),
                (700, 520), (560, 470), (465, 555), (350, 475),
                (215, 530), (145, 405), (235, 330), (125, 275)
            ],
        },
    }

    SENSOR_ANGLES = (-75, -50, -25, 0, 25, 50, 75)
    SENSOR_MAX = 150

    def __init__(self, screen, track_number=1):
        self.screen = screen
        self.set_track(track_number)

    @classmethod
    def available_tracks(cls):
        return cls.TRACKS

    def set_track(self, track_number):
        if track_number not in self.TRACKS:
            track_number = 1

        self.track_number = track_number
        self.track_data = self.TRACKS[track_number]
        self.points = self.track_data["points"]
        self.road_width = self.track_data["width"]

        self.samples = self._build_samples(self.points, 24)

        # Start/finish is placed one quarter of the way around the circuit,\n        # on the centerline rather than at the original corner.
        self.start_index = len(self.samples) // 4
        self.finish_center = self.samples[self.start_index]
        self.start_progress = self.start_index / len(self.samples)

        self.start_angle = self.get_desired_heading(
            *self.finish_center
        )

        self.finish_line = self._build_finish_line()
        self._background = self._render_background()

    def _build_samples(self, points, steps):
        samples = []
        count = len(points)

        for i in range(count):
            p0 = points[(i - 1) % count]
            p1 = points[i]
            p2 = points[(i + 1) % count]
            p3 = points[(i + 2) % count]

            for s in range(steps):
                t = s / steps
                t2 = t * t
                t3 = t2 * t

                x = 0.5 * (
                    (2 * p1[0])
                    + (-p0[0] + p2[0]) * t
                    + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2
                    + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3
                )

                y = 0.5 * (
                    (2 * p1[1])
                    + (-p0[1] + p2[1]) * t
                    + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2
                    + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3
                )

                samples.append((x, y))

        return samples

    def _nearest_sample(self, x, y):
        best_index = 0
        best_distance_sq = float("inf")

        for i, (sx, sy) in enumerate(self.samples):
            dx = x - sx
            dy = y - sy
            distance_sq = dx * dx + dy * dy

            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_index = i

        return best_index, math.sqrt(best_distance_sq)

    def get_nearest_progress(self, x, y):
        index, distance = self._nearest_sample(x, y)
        return index / len(self.samples), distance

    def get_lateral_offset(self, x, y):
        """Signed distance from the nearest centerline point."""
        index, _ = self._nearest_sample(x, y)
        cx, cy = self.samples[index]
        px, py = self.samples[(index - 2) % len(self.samples)]
        nx, ny = self.samples[(index + 2) % len(self.samples)]
        tangent_x = nx - px
        tangent_y = ny - py
        length = math.hypot(tangent_x, tangent_y) or 1.0
        tangent_x /= length
        tangent_y /= length
        offset_x = x - cx
        offset_y = y - cy
        return tangent_x * offset_y - tangent_y * offset_x

    def get_progress_distance(self, x, y):
        return self.get_nearest_progress(x, y)

    def get_desired_heading(self, x, y):
        index, _ = self._nearest_sample(x, y)
        next_index = (index + 3) % len(self.samples)

        x1, y1 = self.samples[index]
        x2, y2 = self.samples[next_index]

        return math.degrees(math.atan2(y2 - y1, x2 - x1))

    def get_start_position(self):
        x, y = self.finish_center

        # Cars start just behind the line, centered on the road.
        angle = self.start_angle
        radians = math.radians(angle)

        start_distance = 42
        x -= math.cos(radians) * start_distance
        y -= math.sin(radians) * start_distance

        return float(x), float(y), float(angle)

    def get_finish_line(self):
        return self.finish_line

    def _build_finish_line(self):
        cx, cy = self.finish_center

        # Road direction at the start/finish point.
        index = self.start_index
        prev_x, prev_y = self.samples[(index - 3) % len(self.samples)]
        next_x, next_y = self.samples[(index + 3) % len(self.samples)]

        dx = next_x - prev_x
        dy = next_y - prev_y
        length = math.hypot(dx, dy) or 1.0

        # Perpendicular normal.
        nx = -dy / length
        ny = dx / length

        half_length = self.road_width * 0.52

        p1 = (
            cx - nx * half_length,
            cy - ny * half_length,
        )
        p2 = (
            cx + nx * half_length,
            cy + ny * half_length,
        )

        return p1, p2

    def get_progress_from_start(self, progress):
        """Return centerline progress measured from the start/finish line."""
        return (progress - self.start_progress) % 1.0

    def is_past_start_line_forward(self, previous_progress, current_progress):
        """Detect a forward crossing of this track's actual start/finish line."""
        previous_relative = self.get_progress_from_start(previous_progress)
        current_relative = self.get_progress_from_start(current_progress)
        return previous_relative > 0.90 and current_relative < 0.10

    def is_on_road(self, x, y):
        """Return True when a point is within half the road width of the centerline."""
        _, nearest_distance = self._nearest_sample(x, y)
        return nearest_distance <= self.road_width / 2

    def _ray_distance(self, x, y, angle):
        radians = math.radians(angle)

        for distance in range(5, self.SENSOR_MAX + 1, 5):
            test_x = x + math.cos(radians) * distance
            test_y = y + math.sin(radians) * distance

            if not self.is_on_road(test_x, test_y):
                return distance

        return self.SENSOR_MAX

    def get_sensor_readings(self, x, y, angle):
        return [
            self._ray_distance(x, y, angle + offset)
            for offset in self.SENSOR_ANGLES
        ]

    def _render_background(self):
        """Build a detailed, static scenery layer once per track for good FPS."""
        rng = random.Random(7400 + self.track_number)
        layer = pygame.Surface((self.WIDTH, self.HEIGHT)).convert()

        # Layered grass base with broad color variation, not noisy per-frame work.
        layer.fill((47, 119, 59))
        tile = 28
        for row, y in enumerate(range(0, self.HEIGHT, tile)):
            for col, x in enumerate(range(0, self.WIDTH, tile)):
                shift = 5 if (row + col) % 2 == 0 else -3
                color = (max(0, 47 + shift), max(0, 119 + shift), max(0, 59 + shift))
                pygame.draw.rect(layer, color, (x, y, tile, tile))
                # Tiny grass tufts are sparse and deterministic.
                if (row * 7 + col * 11) % 3 == 0:
                    gx, gy = x + 10 + (col * 13) % 22, y + 9 + (row * 7) % 24
                    pygame.draw.line(layer, (64, 137, 69), (gx, gy), (gx + 3, gy - 4), 1)

        # Scenic ponds in fixed open areas; decorative only, no physics impact.
        ponds = [
            ([(28, 235), (48, 202), (92, 194), (123, 218), (116, 250), (73, 266), (37, 255)], (43, 133, 160)),
            ([(870, 72), (920, 65), (958, 91), (947, 124), (900, 131), (873, 108)], (39, 128, 157)),
            ([(855, 590), (887, 566), (930, 577), (963, 610), (942, 642), (894, 638), (866, 617)], (43, 135, 160)),
        ]
        for pts, water in ponds:
            if all(not self.is_on_road(x, y) for x, y in pts):
                pygame.draw.polygon(layer, (31, 83, 91), pts)
                pygame.draw.polygon(layer, water, [(x, y - 3) for x, y in pts])
                for j in range(3):
                    pygame.draw.arc(layer, (111, 190, 194), (min(x for x, y in pts)+8+j*4, min(y for x, y in pts)+8+j*4, 30+j*8, 14+j*4), 0.2, 2.7, 1)

        # Track geometry and road shoulders.
        left_edge, right_edge = [], []
        half = self.road_width / 2
        for i, (x, y) in enumerate(self.samples):
            px, py = self.samples[(i - 2) % len(self.samples)]
            nx, ny = self.samples[(i + 2) % len(self.samples)]
            dx, dy = nx - px, ny - py
            length = math.hypot(dx, dy) or 1.0
            normal_x, normal_y = -dy / length, dx / length
            left_edge.append((int(x + normal_x * half), int(y + normal_y * half)))
            right_edge.append((int(x - normal_x * half), int(y - normal_y * half)))

        polygon = left_edge + list(reversed(right_edge))
        # Soft wide shoulder, dark asphalt, then a lighter racing surface.
        pygame.draw.lines(layer, (30, 72, 39), True, self.samples, int(self.road_width + 28))
        pygame.draw.lines(layer, (203, 199, 174), True, self.samples, int(self.road_width + 15))
        pygame.draw.lines(layer, (48, 50, 55), True, self.samples, int(self.road_width + 7))
        pygame.draw.lines(layer, (69, 71, 75), True, self.samples, int(self.road_width))

        # Fine asphalt texture and faint rubber racing line.
        for i in range(0, len(self.samples), 5):
            x, y = self.samples[i]
            pygame.draw.circle(layer, (78, 80, 83), (int(x), int(y)), 1)
        pygame.draw.lines(layer, (54, 56, 60), True, self.samples, max(5, int(self.road_width * 0.24)))
        pygame.draw.lines(layer, (91, 92, 93), True, self.samples, 2)

        # Kerbs: short alternating red/ivory segments on both edges.
        kerb_width = max(4, int(self.road_width * 0.085))
        for edge in (left_edge, right_edge):
            for i in range(0, len(edge), 8):
                a, b = edge[i], edge[(i + 5) % len(edge)]
                color = (225, 55, 48) if (i // 8) % 2 == 0 else (241, 236, 215)
                pygame.draw.line(layer, color, a, b, kerb_width)
        pygame.draw.lines(layer, (242, 239, 222), True, left_edge, 2)
        pygame.draw.lines(layer, (242, 239, 222), True, right_edge, 2)

        # Center dashes.
        for i in range(0, len(self.samples), 27):
            a = self.samples[i]
            b = self.samples[(i + 7) % len(self.samples)]
            pygame.draw.line(layer, (173, 174, 169), (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 2)

        # Generate trees, shrubs, boulders, flowers, signs and tire stacks only
        # in open ground, away from the road. They are static, so runtime cost is low.
        placed = []
        attempts = 0
        while len(placed) < 235 and attempts < 5000:
            attempts += 1
            x = rng.randint(22, self.WIDTH - 22)
            y = rng.randint(22, self.HEIGHT - 22)
            if self.is_on_road(x, y):
                continue
            # Keep a little clear space around the road and avoid dense clumping.
            _, distance = self._nearest_sample(x, y)
            if distance < self.road_width / 2 + 22:
                continue
            if any((x-px)**2 + (y-py)**2 < 18**2 for px, py, _ in placed):
                continue
            kind = rng.choices(("tree", "bush", "rock", "flower"), (42, 30, 18, 10))[0]
            placed.append((x, y, kind))

        for x, y, kind in placed:
            if kind == "tree":
                r = rng.randint(13, 21)
                pygame.draw.ellipse(layer, (31, 72, 38), (x-r+3, y-r+5, 2*r, 2*r))
                pygame.draw.circle(layer, (27, 83, 43), (x, y), r)
                pygame.draw.circle(layer, rng.choice(((39, 105, 49), (49, 119, 54), (58, 129, 61))), (x-3, y-4), max(4, r-3))
                pygame.draw.circle(layer, (94, 151, 75), (x-5, y-6), max(2, r//3))
            elif kind == "bush":
                r = rng.randint(6, 11)
                pygame.draw.circle(layer, (34, 88, 43), (x, y+2), r+1)
                pygame.draw.circle(layer, rng.choice(((48, 112, 50), (61, 127, 57), (39, 99, 47))), (x, y), r)
                pygame.draw.circle(layer, (91, 146, 67), (x-2, y-2), max(2, r//3))
            elif kind == "rock":
                r = rng.randint(5, 10)
                pygame.draw.ellipse(layer, (74, 78, 75), (x-r, y-r//2, 2*r, r+3))
                pygame.draw.ellipse(layer, (123, 125, 116), (x-r+2, y-r//2, r, max(2, r//2)))
            else:
                pygame.draw.circle(layer, (242, 216, 103), (x, y), 3)
                pygame.draw.circle(layer, (230, 231, 211), (x+4, y+2), 2)

        # Simple safety barriers, marshal posts and advertising boards near straights.
        for i in range(10, len(self.samples), max(1, len(self.samples)//22)):
            x, y = self.samples[i]
            px, py = self.samples[(i-2) % len(self.samples)]
            nx, ny = self.samples[(i+2) % len(self.samples)]
            dx, dy = nx-px, ny-py
            length = math.hypot(dx, dy) or 1
            ox, oy = -dy/length, dx/length
            side = -1 if (i//10) % 2 else 1
            bx, by = int(x + ox * (half + 19) * side), int(y + oy * (half + 19) * side)
            if 10 < bx < self.WIDTH-10 and 10 < by < self.HEIGHT-10:
                pygame.draw.rect(layer, (36, 39, 42), (bx-7, by-4, 14, 8), border_radius=2)
                pygame.draw.rect(layer, (218, 225, 220), (bx-6, by-3, 4, 6))
                pygame.draw.rect(layer, (40, 154, 190), (bx-1, by-3, 6, 6))

        # Stronger visual landmarks: forest groves, landscaped flower beds,
        # service buildings and spectator stands. Decorative only; no physics impact.
        for cx, cy, rx, ry in ((85, 95, 54, 30), (910, 300, 38, 65),
                               (900, 485, 42, 34), (90, 570, 48, 32),
                               (490, 625, 65, 22), (680, 70, 45, 24)):
            if not self.is_on_road(cx, cy):
                pygame.draw.ellipse(layer, (39, 99, 48), (cx-rx, cy-ry, rx*2, ry*2))
                for k in range(9):
                    tx = cx + rng.randint(-rx+5, rx-5)
                    ty = cy + rng.randint(-ry+5, ry-5)
                    if not self.is_on_road(tx, ty):
                        rr = rng.randint(7, 13)
                        pygame.draw.ellipse(layer, (28, 67, 35), (tx-rr+3, ty-rr+4, rr*2, rr*2))
                        pygame.draw.circle(layer, rng.choice(((37, 91, 43), (48, 113, 49), (62, 128, 58))), (tx, ty), rr)
                        pygame.draw.circle(layer, (100, 157, 77), (tx-3, ty-4), max(3, rr//3))

        # Grandstands are placed only in clear, off-road areas.
        for rect in ((35, 390, 104, 34), (820, 24, 112, 30), (785, 635, 118, 28)):
            bx, by, bw, bh = rect
            cx, cy = bx + bw//2, by + bh//2
            if all(not self.is_on_road(px, py) for px, py in
                   ((bx, by), (bx+bw, by), (bx, by+bh), (bx+bw, by+bh), (cx, cy))):
                pygame.draw.rect(layer, (30, 34, 39), (bx+3, by+5, bw, bh), border_radius=3)
                pygame.draw.rect(layer, (177, 184, 184), (bx, by, bw, bh), border_radius=3)
                pygame.draw.rect(layer, (53, 61, 70), (bx+4, by+4, bw-8, bh-8))
                for row in range(2):
                    for seat in range(0, bw-12, 7):
                        color = rng.choice(((226, 65, 56), (54, 147, 218), (243, 193, 65), (225, 229, 218)))
                        pygame.draw.circle(layer, color, (bx+7+seat, by+8+row*9), 2)

        # More visible perimeter fence posts around selected straight sections.
        for i in range(0, len(self.samples), 18):
            x, y = self.samples[i]
            px, py = self.samples[(i-2) % len(self.samples)]
            nx, ny = self.samples[(i+2) % len(self.samples)]
            dx, dy = nx-px, ny-py
            length = math.hypot(dx, dy) or 1
            ox, oy = -dy/length, dx/length
            for side in (-1, 1):
                fx = int(x + ox*(half+25)*side)
                fy = int(y + oy*(half+25)*side)
                if 8 < fx < self.WIDTH-8 and 8 < fy < self.HEIGHT-8:
                    if not self.is_on_road(fx, fy):
                        pygame.draw.circle(layer, (224, 220, 197), (fx, fy), 2)

        # Start/finish is drawn into the cached background too.
        self._draw_finish_line_to(layer)
        return layer

    def draw(self):
        # Blit pre-rendered scenery; no random work or terrain drawing during racing.
        self.screen.blit(self._background, (0, 0))

    def _draw_finish_line_to(self, target):
        p1, p2 = self.finish_line
        pygame.draw.line(target, (245, 245, 245), p1, p2, 12)
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        for i in range(8):
            t1, t2 = i / 8, (i + 1) / 8
            a = (int(p1[0] + dx*t1), int(p1[1] + dy*t1))
            b = (int(p1[0] + dx*t2), int(p1[1] + dy*t2))
            pygame.draw.line(target, (22, 23, 26) if i % 2 == 0 else (248, 248, 240),
                             a, b, 12)
        cx, cy = self.finish_center
        font = pygame.font.SysFont("Arial", 14, bold=True)
        label = font.render("START / FINISH", True, (255, 255, 255))
        plate = pygame.Surface((label.get_width()+16, label.get_height()+8), pygame.SRCALPHA)
        pygame.draw.rect(plate, (16, 20, 25, 205), plate.get_rect(), border_radius=5)
        plate.blit(label, (8, 4))
        target.blit(plate, (int(cx)-plate.get_width()//2,
                                 int(cy)-int(self.road_width//2)-plate.get_height()-8))

    def draw_finish_line(self):
        # The finish line is already part of the cached track background.
        if self._background is not None:
            self.screen.blit(self._background, (0, 0))

    def draw_sensors(self, x, y, angle, sensors):
        for offset, distance in zip(self.SENSOR_ANGLES, sensors):
            radians = math.radians(angle + offset)
            end_x = x + math.cos(radians) * distance
            end_y = y + math.sin(radians) * distance
            color = (70, 225, 255) if distance > 55 else (255, 190, 70)
            pygame.draw.line(self.screen, color, (int(x), int(y)),
                             (int(end_x), int(end_y)), 2)
            pygame.draw.circle(self.screen, (245, 250, 255),
                               (int(end_x), int(end_y)), 3)
