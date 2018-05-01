"""Discrete event simulator."""

# Standard libs.
import heapq
import logging


class DiscreteEvent(object):
    """Base class for a discrete event."""

    def __init__(self, handler, timestamp, **handlerKwargs):
        self.timestamp = timestamp
        self.handler = handler
        self.handlerKwargs = handlerKwargs

    def __str__(self):
        """Returns a description of the event.

        Returns:
            String description of the event. If the event has been processed,
            the description will include details about the event outcome.
        """
        return 'T={0:.2f}, {1}'.format(
            self.timestamp, self.handler.__name__)

    def __cmp__(self, other):
         return cmp(self.timestamp, other.timestamp)


class DiscreteEventSimulationEngine(object):
    """Discrete event simulation engine."""

    def __init__(self):
        # Initialize simulation time.
        self.simTime = 0
        # The FEL is a timestamp-based priority queue.
        self.FEL = []

    def schedule(self, event):
        """Schedules a discrete event in the FEL.

        Args:
            event: An instance of DiscreteEvent.
        """
        heapq.heappush(self.FEL, event)

    def runSimulation(self, maxEvents=float('inf')):
        """Processes all events in the FEL.

        Args:
            maxEvents: Maximum number of events to process. If unspecified,
                all events in the FEL will be processed.
        """
        numEventsProcessed = 0
        while len(self.FEL) > 0 and numEventsProcessed < maxEvents:
            event = heapq.heappop(self.FEL)
            logging.debug(event)
            # Update simulation time.
            self.simTime = event.timestamp
            # Process the event.
            event.handler(self, **event.handlerKwargs)
            numEventsProcessed += 1
        logging.info('Processed %d events.' % numEventsProcessed)

    def currentTime(self):
        """Returns the current simulation time.

        Returns:
            Positive number representing the current simulation time.
        """
        return self.simTime
