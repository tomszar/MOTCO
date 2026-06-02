## ADDED Requirements

### Requirement: Bridge rejects out-of-range seeds before invoking R

The InterSIM bridge SHALL validate that the supplied seed is within
R's signed-32-bit range `[-2³¹, 2³¹ − 1]` and raise a clear Python
error before launching the R subprocess when it is not.

#### Scenario: Out-of-range positive seed is rejected with a clear Python error

- **WHEN** a caller invokes the bridge with a seed greater than
  `2³¹ − 1`
- **THEN** the bridge raises an `InterSIMError` (or subclass) whose
  message names the offending value and the accepted range, without
  starting the R subprocess

#### Scenario: Out-of-range negative seed is rejected with a clear Python error

- **WHEN** a caller invokes the bridge with a seed less than `-2³¹`
- **THEN** the bridge raises an `InterSIMError` (or subclass) whose
  message names the offending value and the accepted range, without
  starting the R subprocess

#### Scenario: In-range seed proceeds normally

- **WHEN** a caller invokes the bridge with a seed in
  `[-2³¹, 2³¹ − 1]`
- **THEN** the bridge launches the R subprocess and the seed reaches
  `set.seed` unchanged
