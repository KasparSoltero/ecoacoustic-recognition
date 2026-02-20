from matplotlib.colors import LinearSegmentedColormap

dusk_colors = [
    (255, 255, 255),
(255, 253, 255),
(255, 251, 255),
(255, 249, 255),
(255, 247, 255),
(255, 245, 255),
(255, 243, 255),
(255, 241, 255),
(255, 239, 255),
(255, 237, 255),
(255, 235, 255),
(255, 233, 255),
(255, 231, 255),
(255, 229, 255),
(255, 227, 255),
(255, 225, 255),
(255, 223, 255),
(255, 221, 255),
(255, 219, 255),
(255, 217, 255),
(255, 215, 255),
(255, 213, 255),
(255, 211, 255),
(255, 209, 255),
(255, 207, 255),
(255, 205, 255),
(255, 203, 255),
(255, 201, 255),
(255, 199, 255),
(255, 197, 255),
(255, 195, 255),
(255, 193, 255),
(255, 191, 255),
(255, 189, 255),
(255, 187, 255),
(255, 185, 255),
(255, 183, 255),
(255, 181, 255),
(255, 179, 255),
(255, 177, 255),
(255, 175, 255),
(255, 173, 255),
(255, 172, 255),
(255, 170, 255),
(255, 168, 255),
(255, 166, 255),
(255, 164, 255),
(255, 162, 255),
(255, 160, 255),
(255, 158, 255),
(255, 156, 255),
(255, 154, 255),
(255, 152, 255),
(255, 150, 255),
(255, 148, 255),
(255, 146, 255),
(255, 144, 255),
(255, 142, 255),
(255, 140, 255),
(255, 138, 255),
(255, 136, 255),
(255, 134, 255),
(255, 132, 255),
(255, 130, 255),
(255, 128, 255),
(253, 126, 255),
(251, 124, 255),
(249, 122, 255),
(247, 120, 255),
(245, 118, 255),
(242, 116, 255),
(241, 114, 255),
(238, 112, 255),
(237, 110, 255),
(235, 108, 255),
(233, 106, 255),
(231, 104, 255),
(229, 102, 255),
(227, 100, 255),
(225, 98, 255),
(223, 96, 255),
(221, 94, 255),
(219, 92, 255),
(217, 90, 255),
(215, 88, 255),
(213, 86, 255),
(211, 84, 255),
(209, 81, 255),
(207, 79, 255),
(205, 77, 255),
(203, 75, 255),
(201, 73, 255),
(199, 71, 255),
(197, 69, 255),
(195, 67, 255),
(193, 65, 255),
(191, 63, 255),
(189, 61, 255),
(187, 59, 255),
(185, 57, 255),
(183, 55, 255),
(181, 53, 255),
(179, 51, 255),
(177, 49, 255),
(175, 47, 255),
(173, 45, 255),
(171, 43, 255),
(169, 41, 255),
(167, 39, 255),
(165, 37, 255),
(163, 35, 255),
(161, 33, 255),
(159, 31, 255),
(157, 29, 255),
(155, 27, 255),
(153, 25, 255),
(151, 23, 255),
(149, 21, 255),
(147, 19, 255),
(145, 17, 255),
(143, 15, 255),
(141, 13, 255),
(138, 11, 255),
(136, 9, 255),
(134, 7, 255),
(132, 5, 255),
(131, 3, 255),
(129, 1, 255),
(126, 0, 254),
(125, 0, 252),
(122, 0, 250),
(121, 0, 248),
(118, 0, 246),
(116, 0, 244),
(115, 0, 242),
(113, 0, 240),
(111, 0, 238),
(109, 0, 236),
(107, 0, 234),
(105, 0, 232),
(102, 0, 230),
(100, 0, 228),
(98, 0, 227),
(97, 0, 225),
(94, 0, 223),
(93, 0, 221),
(91, 0, 219),
(89, 0, 217),
(87, 0, 215),
(84, 0, 213),
(83, 0, 211),
(81, 0, 209),
(79, 0, 207),
(77, 0, 205),
(75, 0, 203),
(73, 0, 201),
(70, 0, 199),
(68, 0, 197),
(66, 0, 195),
(64, 0, 193),
(63, 0, 191),
(61, 0, 189),
(59, 0, 187),
(57, 0, 185),
(54, 0, 183),
(52, 0, 181),
(51, 0, 179),
(49, 0, 177),
(47, 0, 175),
(44, 0, 174),
(42, 0, 172),
(40, 0, 170),
(39, 0, 168),
(37, 0, 166),
(34, 0, 164),
(33, 0, 162),
(31, 0, 160),
(29, 0, 158),
(27, 0, 156),
(25, 0, 154),
(22, 0, 152),
(20, 0, 150),
(18, 0, 148),
(17, 0, 146),
(14, 0, 144),
(13, 0, 142),
(11, 0, 140),
(9, 0, 138),
(6, 0, 136),
(4, 0, 134),
(2, 0, 132),
(0, 0, 130),
(0, 0, 128),
(0, 0, 126),
(0, 0, 124),
(0, 0, 122),
(0, 0, 120),
(0, 0, 118),
(0, 0, 116),
(0, 0, 114),
(0, 0, 112),
(0, 0, 110),
(0, 0, 108),
(0, 0, 106),
(0, 0, 104),
(0, 0, 102),
(0, 0, 100),
(0, 0, 98),
(0, 0, 96),
(0, 0, 94),
(0, 0, 92),
(0, 0, 90),
(0, 0, 88),
(0, 0, 86),
(0, 0, 83),
(0, 0, 81),
(0, 0, 79),
(0, 0, 77),
(0, 0, 75),
(0, 0, 73),
(0, 0, 71),
(0, 0, 69),
(0, 0, 67),
(0, 0, 65),
(0, 0, 63),
(0, 0, 61),
(0, 0, 59),
(0, 0, 57),
(0, 0, 55),
(0, 0, 53),
(0, 0, 51),
(0, 0, 49),
(0, 0, 47),
(0, 0, 45),
(0, 0, 43),
(0, 0, 41),
(0, 0, 39),
(0, 0, 37),
(0, 0, 35),
(0, 0, 33),
(0, 0, 31),
(0, 0, 29),
(0, 0, 26),
(0, 0, 24),
(0, 0, 22),
(0, 0, 20),
(0, 0, 18),
(0, 0, 16),
(0, 0, 14),
(0, 0, 12),
(0, 0, 10),
(0, 0, 8),
(0, 0, 6),
(0, 0, 4),
(0, 0, 2),
(0, 0, 0),
]
dusk_colors.reverse()
dusk_colormap = LinearSegmentedColormap.from_list("dusk", [tuple(color/255 for color in c) for c in dusk_colors])
custom_color_maps = {
    'dusk': dusk_colormap,
    'teal': '#00FFFF',
    'rotary': [
        '#ff0000', 
        '#59ff00', 
        '#001eff', 
        '#fb00ff',
        '#00aeff',
        '#ff8c00', 
        '#6f00ff', 
        '#e6ff00', 
        '#00ffc3', 
        ]
}


def hex_to_rgb(hex_color):
    # Remove the '#' if present
    hex_color = hex_color.lstrip('#')
    # Convert to RGB
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

import colorsys
import math

def generate_rainbow_colors(num_colors: int) -> list[str]:
    """
    Generates a list of hex color codes interpolating a smooth rainbow.

    The rainbow starts at Red (hue=0.0) and ends near Violet (hue=0.75),
    maintaining full saturation and standard lightness.

    Args:
        num_colors: The number of distinct colors to generate (integer).
                    Must be non-negative.

    Returns:
        A list of hex color strings (e.g., '#FF0000').
        Returns an empty list if num_colors is 0 or negative.
        Returns ['#FF0000'] if num_colors is 1.

    Raises:
        TypeError: If num_colors is not an integer.
        ValueError: If num_colors is negative (handled by returning []).
    """
    if not isinstance(num_colors, int):
        raise TypeError("Input must be an integer.")
    if num_colors <= 0:
        return []

    hex_colors = []

    # Define HSL parameters for the rainbow
    saturation = 1.0  # Full saturation for vibrant colors
    lightness = 0.5   # Standard lightness (0.0=black, 1.0=white)
    start_hue = 0.0   # Red
    # End hue slightly before red again (e.g., 270 degrees / 360 = 0.75 for Violet)
    # Adjust this value (0.0 to 1.0) to change the end color of the rainbow
    end_hue = 0.75    # Violet

    if num_colors == 1:
        # Special case for a single color: return Red
        rgb_float = colorsys.hls_to_rgb(start_hue, lightness, saturation)
        # Convert float (0-1) to int (0-255)
        rgb_int = tuple(max(0, min(255, int(round(c * 255)))) for c in rgb_float)
        # Format as hex string and return
        return [f"#{rgb_int[0]:02x}{rgb_int[1]:02x}{rgb_int[2]:02x}".upper()]

    # Generate colors for num_colors > 1
    for i in range(num_colors):
        # Calculate the current hue, interpolating linearly between start and end hue
        # We divide by (num_colors - 1) to ensure the last color hits end_hue exactly
        hue = start_hue + (end_hue - start_hue) * i / (num_colors - 1)

        # Convert HSL to RGB (results are floats between 0.0 and 1.0)
        rgb_float = colorsys.hls_to_rgb(hue, lightness, saturation)

        # Convert RGB floats to integer values (0-255)
        # Use round() for better accuracy and clamp values between 0 and 255
        r = max(0, min(255, int(round(rgb_float[0] * 255))))
        g = max(0, min(255, int(round(rgb_float[1] * 255))))
        b = max(0, min(255, int(round(rgb_float[2] * 255))))

        # Format the RGB tuple into a hex color string (e.g., #FF0000)
        # Use :02x formatting to ensure two digits for each component (padding with 0 if needed)
        # Use .upper() for standard uppercase hex codes
        hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
        hex_colors.append(hex_color)

    return hex_colors