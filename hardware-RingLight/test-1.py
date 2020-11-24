import RPi.GPIO as gpio 
import time
import neopixel_spi as neo
import board

NUM_PIXELS = 16
PIXEL_ORDER = neo.GRB
COLORS = (0xFF0000, 0x00FF00, 0x0000FF)
DELAY = 0.1
 
spi = board.SPI()
 
pixels = neo.NeoPixel_SPI(
    spi, NUM_PIXELS, pixel_order=PIXEL_ORDER, auto_write=False
)
 

gpio.setmode(gpio.TEGRA_SOC)

while True:
    try:
        for color in COLORS:
            for i in range(NUM_PIXELS):
                pixels[i] = color
                pixels.show()
                time.sleep(DELAY)
                pixels.fill(0)
    except KeyboardInterrupt:
        pixels.fill(0xffffff)
        pixels.show()
        time.sleep(1)
        pixels.fill(0)
        pixels.show()
        gpio.cleanup()
        break
