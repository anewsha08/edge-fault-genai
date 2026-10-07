# firmware/

Cortex-M code for the sensor node.

## Phase 1 target
- Timer-interrupt sampling of the accelerometer at a fixed rate (start at 1 kHz)
- DMA or ring buffer, no dropped samples
- 256-sample windows, 50% overlap
- Raw window streaming over UART for plotting on the PC

## Suggested layout
```
firmware/
  CMakeLists.txt
  src/
    main.c
    sensor.c / sensor.h      # accelerometer driver (I2C or SPI)
    sampler.c / sampler.h    # timer ISR + ring buffer
    uart_proto.c / .h        # framing for messages to the Pi
  docs/
    wiring.md                # pin table + photo
```

## Measurements to record
| Item | Value |
|---|---|
| Sample rate (configured / measured) | |
| Worst-case sample jitter | |
| CPU load during sampling | |
| RAM / flash used | |
