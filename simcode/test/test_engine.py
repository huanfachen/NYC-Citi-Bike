"""Tests for the DiscreteEventSimulationEngine."""

import unittest
import simcode.src.engine as engine


def MockEvent(simEngine, **kwargs):
    """Event handler used for unit testing purposes."""
    kwargs['data']['processed'] = True


class TestDiscreteEventSimulationEngine(unittest.TestCase):
    """Unit tests for DiscreteEventSimulationEngine."""

    def setUp(self):
        """Sets up before each test method."""
        self.simEngine = engine.DiscreteEventSimulationEngine()

    def test_schedule(self):
        """Tests scheduling of events."""
        # FEL is empty before adding events.
        self.assertEqual(0, len(self.simEngine.FEL))

        # An event is scheduled in the FEL.
        testEventTimestamp = 10
        testEvent = engine.DiscreteEvent(
            MockEvent, testEventTimestamp, data={'processed': False})
        self.simEngine.schedule(testEvent)
        self.assertEqual(
            [testEvent], self.simEngine.FEL)

        # An earlier event is scheduled at the front of the FEL.
        earlierTestEvent = engine.DiscreteEvent(
            MockEvent, testEventTimestamp - 1, data={'processed': False})
        self.simEngine.schedule(earlierTestEvent)
        self.assertEqual(
            [earlierTestEvent, testEvent],
            self.simEngine.FEL)

        # An later event is scheduled at the end of the FEL.
        laterTestEvent = engine.DiscreteEvent(
            MockEvent, testEventTimestamp + 1, data={'processed': False})
        self.simEngine.schedule(laterTestEvent)
        self.assertEqual(
            [earlierTestEvent, testEvent, laterTestEvent],
            self.simEngine.FEL)

    def test_runSimulation(self):
        """Tests execution of simulation events."""
        # The simulation time is at zero.
        self.assertEqual(0, self.simEngine.currentTime())

        # The FEL contains events.
        testEventTimestamp = 10
        testEventData = {'processed': False}
        testEvent = engine.DiscreteEvent(
            MockEvent, testEventTimestamp, data=testEventData)
        self.simEngine.schedule(testEvent)

        earlierTestEventData = {'processed': False}
        earlierTestEvent = engine.DiscreteEvent(
            MockEvent, testEventTimestamp - 1, data=earlierTestEventData)
        self.simEngine.schedule(earlierTestEvent)
        
        # The events are unprocessed.
        self.assertEqual(2, len(self.simEngine.FEL))
        self.assertEqual(False, testEventData['processed'])
        self.assertEqual(False, earlierTestEventData['processed'])

        # The simulation runs.
        self.simEngine.runSimulation()

        # All events in the FEL were processed.
        self.assertEqual(0, len(self.simEngine.FEL))
        self.assertEqual(True, testEventData['processed'])
        self.assertEqual(True, earlierTestEventData['processed'])
        # The simulation time has been updated.
        self.assertEqual(testEventTimestamp, self.simEngine.currentTime())

    def test_runSimulation_maxEvents(self):
        """Tests execution of simulation events with max_event parameter."""
        # The FEL contains multiple events.
        testEventTimestamp = 10
        testEventData = {'processed': False}
        testEvent = engine.DiscreteEvent(
            MockEvent, testEventTimestamp, data=testEventData)
        self.simEngine.schedule(testEvent)

        earlierTestEventData = {'processed': False}
        earlierTestEvent = engine.DiscreteEvent(
            MockEvent, testEventTimestamp - 1, data=earlierTestEventData)
        self.simEngine.schedule(earlierTestEvent)

        # The events are unprocessed.
        self.assertEqual(2, len(self.simEngine.FEL))
        self.assertEqual(False, testEventData['processed'])
        self.assertEqual(False, earlierTestEventData['processed'])

        # The simulation runs with max_events parameter.
        self.simEngine.runSimulation(maxEvents=1)

        # Only max_event events were processed.
        self.assertEqual([testEvent], self.simEngine.FEL)
        self.assertEqual(False, testEventData['processed'])
        self.assertEqual(True, earlierTestEventData['processed'])
        # The simulation time has been updated.
        self.assertEqual(
            earlierTestEvent.timestamp, self.simEngine.currentTime())


if __name__ == '__main__':
    unittest.main()
