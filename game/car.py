import math
import pygame


class Car:
    """Top-down racing car with a shaded body, wheels, glass and lights."""

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.angle = 90
        self.speed = 0.0
        self.acceleration = 0.15
        self.max_speed = 5.0
        self.friction = 0.97
        self.rotation_speed = 3.0
        self.width = 24
        self.height = 40
        self.color = (40, 130, 255)
        self._sprite_cache = None
        self._sprite_color = None

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.speed += self.acceleration
        if keys[pygame.K_DOWN]:
            self.speed -= self.acceleration
        if keys[pygame.K_LEFT]:
            self.angle -= self.rotation_speed
        if keys[pygame.K_RIGHT]:
            self.angle += self.rotation_speed
        self.speed = max(-self.max_speed * 0.5, min(self.max_speed, self.speed))

    def apply_ai_action(self, action):
        if action == 0:
            self.speed += self.acceleration
        elif action == 1:
            self.angle -= self.rotation_speed
        elif action == 2:
            self.angle += self.rotation_speed
        elif action == 3:
            self.speed *= 0.8
        self.speed = max(-self.max_speed * 0.5, min(self.max_speed, self.speed))

    def update(self):
        self.speed *= self.friction
        radians = math.radians(self.angle)
        self.x += math.cos(radians) * self.speed
        self.y += math.sin(radians) * self.speed

    def _build_sprite(self):
        w, h = 42, 58
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        cx, cy = w // 2, h // 2

        # Soft shadow and rubber wheels.
        pygame.draw.ellipse(s, (0, 0, 0, 75), (5, 7, 34, 48))
        for wx in (5, 29):
            pygame.draw.rect(s, (15, 17, 21), (wx, 13, 8, 14), border_radius=3)
            pygame.draw.rect(s, (15, 17, 21), (wx, 34, 8, 14), border_radius=3)

        # Main shell with darker lower edge.
        body = pygame.Rect(8, 4, 26, 50)
        dark = tuple(max(0, int(c * 0.52)) for c in self.color)
        pygame.draw.rect(s, dark, body.inflate(2, 1), border_radius=9)
        pygame.draw.rect(s, self.color, body, border_radius=8)
        pygame.draw.rect(s, tuple(min(255, int(c * 1.28 + 18)) for c in self.color),
                         (11, 7, 20, 12), border_radius=5)

        # Front nose, windshield, rear glass.
        pygame.draw.polygon(s, tuple(min(255, c + 35) for c in self.color),
                            [(21, 1), (12, 12), (30, 12)])
        pygame.draw.polygon(s, (42, 65, 82), [(13, 15), (29, 15), (27, 25), (15, 25)])
        pygame.draw.polygon(s, (90, 190, 210), [(15, 16), (27, 16), (26, 23), (16, 23)])
        pygame.draw.polygon(s, (25, 37, 49), [(15, 33), (27, 33), (29, 41), (13, 41)])
        pygame.draw.line(s, (190, 235, 245), (16, 17), (20, 17), 1)

        # Racing stripe and rear wing.
        pygame.draw.line(s, (235, 240, 245), (21, 27), (21, 33), 2)
        pygame.draw.rect(s, (28, 30, 35), (9, 47, 24, 4), border_radius=2)
        pygame.draw.rect(s, (255, 70, 65), (12, 44, 5, 3), border_radius=1)
        pygame.draw.rect(s, (255, 70, 65), (25, 44, 5, 3), border_radius=1)
        pygame.draw.circle(s, (255, 248, 185), (13, 7), 2)
        pygame.draw.circle(s, (255, 248, 185), (29, 7), 2)
        pygame.draw.rect(s, (235, 240, 245), (10, 27, 2, 7), border_radius=1)
        pygame.draw.rect(s, (235, 240, 245), (30, 27, 2, 7), border_radius=1)
        return s

    def draw(self, screen):
        if self._sprite_cache is None or self._sprite_color != self.color:
            self._sprite_cache = self._build_sprite()
            self._sprite_color = self.color
        # The sprite faces upward at zero rotation; angle 90 points right.
        rotated = pygame.transform.rotate(self._sprite_cache, -self.angle - 90)
        rect = rotated.get_rect(center=(round(self.x), round(self.y)))
        screen.blit(rotated, rect)
