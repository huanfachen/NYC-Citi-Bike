"""Tests for the Citi Bike Sharing simulation application."""

# Standard libs.
import unittest

# Third-party libs.
import numpy as np

# App libs.
import simcode.src.engine as engine
import simcode.src.nycbike as nycbike


class TestBikeSharingSimulation(unittest.TestCase):
    """Unit tests for Citi Bike Sharing simulation."""

    # Trip count data used in tests. Time frames are quarterly.
    # Entry [i][j] represents the number of trips at station i in time frame j.
    TEST_TRIP_COUNT_DATA = [
        [4, 1, 2, 3],
        [5, 1, 2, 3],
        [3, 1, 2, 3],
    ]
    # Trip durations used in tests. Units are minutes.
    # Entry [i][j] is the average trip duration from station i to station j.
    TEST_TRIP_DURATIONS = np.array([
        [0, 0.25, 0.1],
        [0.65, 0, 0.8],
        [0.1, 1.2, 0],
    ])
    # Destination probabilites used in tests.
    # Entry [i][j][k] is the probability of choosing station k as the
    # destination from station i during time frame j.
    TEST_DEST_PROBS = np.array([    
        [[0.0, 0.0, 1.0], [0.0, 0.4, 0.6], [0.0, 0.4, 0.6], [0.0, 0.4, 0.6]],
        [[0.3, 0.0, 0.7], [0.3, 0.0, 0.7], [0.3, 0.0, 0.7], [0.3, 0.0, 0.7]],
        [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.0]],
    ])
    # Number of stations used in tests.
    TEST_NUM_STATIONS = len(TEST_TRIP_COUNT_DATA)

    # Initial distribution of bikes to stations used in tests.
    TEST_INITIAL_BIKE_DISTRIBUTION = np.ones(TEST_NUM_STATIONS) * 15


    def setUp(self):
        """Sets up before each test method."""
        self.simEngine = engine.DiscreteEventSimulationEngine()

    def _initEntities(self, numStations):
        """Helper method used to initialize station entities."""
        stations = []
        pickupQueues = []
        dropoffQueues = []
        for stationID in range(numStations):
            stations.append(nycbike.Station(stationID, 30, 15))
            pickupQueues.append(nycbike.Queue())
            dropoffQueues.append(nycbike.Queue())
        return stations, pickupQueues, dropoffQueues

    def _initGlobalData(self, numStations, initEntities=True):
        # Global data structures.
        stations = []
        pickupQueues = []
        dropoffQueues = []
        if initEntities:
            stations, pickupQueues, dropoffQueues = self._initEntities(
                self.TEST_NUM_STATIONS)

        # Simulation statistics used in tests.
        statistics = {
            'Revenue': 0,
            'TimeWaitForDropoff': np.zeros(numStations),
            'TimeWaitForCycle': np.zeros(numStations),
            'CustomersLost': np.zeros(numStations),
            'BikesLost': 0,
            'IdleTime': np.zeros(numStations),
        }

        # Simulation global data used in tests.
        TEST_GLOBAL_DATA = {
            # Entities.
            'stations': stations,
            'pickupQueues': pickupQueues,
            'dropoffQueues': dropoffQueues,
            # Citi bike dataset statistics.
            'arrivalTimes': self.TEST_TRIP_COUNT_DATA,
            'tripDurations': self.TEST_TRIP_DURATIONS,
            'destinationP': self.TEST_DEST_PROBS,
            # Simulation statistics.
            'statistics': statistics,
            # Constants.
            'bikeLossProb': 0.0,  # use zero for testing
        }
        return TEST_GLOBAL_DATA

    def test_initializeEvent(self):
        """Tests the Initialize event."""
        self.globalData = self._initGlobalData(
            self.TEST_NUM_STATIONS, initEntities=False)

        # The Initialize event is scheduled (but not yet processed).
        initEvent = engine.DiscreteEvent(
            nycbike.Initialize, -1, globalData=self.globalData,
            initialDistribution=self.TEST_INITIAL_BIKE_DISTRIBUTION,
            racksPerStation=30)
        self.simEngine.schedule(initEvent)

        # Stations are not yet initialized.
        self.assertEqual([], self.globalData['stations'])
        self.assertEqual([], self.globalData['pickupQueues'])
        self.assertEqual([], self.globalData['dropoffQueues'])
        # Arrival events are not yet scheduled.
        self.assertEqual([initEvent], self.simEngine.FEL)

        # The Initialize event is processed.
        self.simEngine.runSimulation(maxEvents=1)

        # Checkout lines/counters are initialized.
        self.assertEqual(
            self.TEST_NUM_STATIONS, len(self.globalData['stations']))
        self.assertEqual(
            self.TEST_NUM_STATIONS, len(self.globalData['pickupQueues']))
        self.assertEqual(
            self.TEST_NUM_STATIONS, len(self.globalData['dropoffQueues']))

        # Arrival events are scheduled.
        self.assertTrue(initEvent not in self.simEngine.FEL)
        self.assertTrue(len(self.simEngine.FEL) > 0)


    def test_arrivalEvent_bikesAvailable(self):
        """Tests the Arrival event when bikes are available."""
        currentTime = 100
        self.simEngine.simTime = currentTime
        self.globalData = self._initGlobalData(
            self.TEST_NUM_STATIONS, initEntities=True)

        # The simulation time is 00:00, so we expect station 2 to be chosen
        # as the destination from station 0 (other stations have zero
        # probability of being the destination).
        self.simEngine.simTime = 0
        expectedDestID = 2

        # There is initially non-zero revenue.
        revenueBefore = 500
        self.globalData['statistics']['Revenue'] = revenueBefore

        # The station has bikes available.
        testStationID = 0
        testStation = self.globalData['stations'][testStationID]
        numBikesBefore = testStation.numBikes
        numRacksBefore = testStation.numRacks
        self.assertTrue(numBikesBefore > 0)

        # A customer has waited too long and should receive a refund.
        testDropoffQueue = self.globalData['dropoffQueues'][testStationID]
        longWaitCustomer = nycbike.Customer()
        longWaitCustomer.startID = testStationID
        longWaitCustomer.startDropoffWait = currentTime - 100  # > REFUND_TIME
        testDropoffQueue.put(longWaitCustomer)

        # The Arrival event is scheduled and processed.
        arrivalEvent = engine.DiscreteEvent(
            nycbike.Arrival, 10, globalData=self.globalData,
            stationID=testStationID)
        self.simEngine.schedule(arrivalEvent)
        self.simEngine.runSimulation(maxEvents=1)

        # The station attributes were updated.
        self.assertEqual(numBikesBefore - 1, testStation.numBikes)
        self.assertEqual(numRacksBefore + 1, testStation.numRacks)

        # The current customer paid for the bike.
        expectedRevenue = revenueBefore + nycbike.TRIP_COST
        # A RideEnd event was scheduled at the destination.
        rideEndEvents = [e for e in self.simEngine.FEL
                         if (e.handler == nycbike.RideEnd)]
        rideEndEvent = rideEndEvents[0]
        # The trip destination is correct.
        ridingCustomer = rideEndEvent.handlerKwargs['customer']
        self.assertEqual(expectedDestID, ridingCustomer.endID)
        # The trip duration is correct.
        expectedRideEndTime = (
            arrivalEvent.timestamp
            + self.TEST_TRIP_DURATIONS[testStationID][expectedDestID])
        self.assertEqual(expectedRideEndTime, rideEndEvent.timestamp)

        # The long-wait customer received a refund and left.
        self.assertFalse(longWaitCustomer in testDropoffQueue)
        expectedRevenue -= nycbike.TRIP_COST

        # The next Arrival event was scheduled for the station.
        scheduledArrivals = [e for e in self.simEngine.FEL
                             if e.handler == nycbike.Arrival]
        self.assertEqual(1, len(scheduledArrivals))
        self.assertEqual(
            testStationID, scheduledArrivals[0].handlerKwargs['stationID'])


    def test_arrivalEvent_noBikesAvailable(self):
        """Tests the Arrival event when no bikes are available."""
        self.globalData = self._initGlobalData(
            self.TEST_NUM_STATIONS, initEntities=True)

        # The station has zero bikes available.
        testStationID = 0
        testStation = self.globalData['stations'][testStationID]
        testStation.numBikes = 0
        numRacksBefore = testStation.numRacks
        # There are zero people waiting for a bike.
        testPickupQueue = self.globalData['pickupQueues'][testStationID]
        self.assertEqual(0, len(testPickupQueue))

        # The Arrival event is scheduled and processed.
        arrivalEvent = engine.DiscreteEvent(
            nycbike.Arrival, 10, globalData=self.globalData,
            stationID=testStationID)
        self.simEngine.schedule(arrivalEvent)
        self.simEngine.runSimulation(maxEvents=1)

        # The customer was put in the pickup waiting queue.
        self.assertEqual(1, len(testPickupQueue))
        # The start of waiting time was recorded.
        customer = testPickupQueue.remove()
        self.assertEqual(arrivalEvent.timestamp, customer.startPickupWait)

        # The station attributes were not updated.
        self.assertEqual(0, testStation.numBikes)
        self.assertEqual(numRacksBefore, testStation.numRacks)

        # Revenue is unchanged.
        self.assertEqual(0, self.globalData['statistics']['Revenue'])

        # The next Arrival event was scheduled for the station.
        scheduledArrivals = [e for e in self.simEngine.FEL
                             if e.handler == nycbike.Arrival]
        self.assertEqual(1, len(scheduledArrivals))
        self.assertEqual(
            testStationID, scheduledArrivals[0].handlerKwargs['stationID'])
        # No other events were scheduled.
        self.assertEqual(1, len(self.simEngine.FEL))  

    def test_rideEndEvent_racksAvailable(self):
        """Tests the RideEnd event when racks are available."""
        currentTime = 100
        self.simEngine.simTime = currentTime
        self.globalData = self._initGlobalData(
            self.TEST_NUM_STATIONS, initEntities=True)

        # The station has racks available.
        testStationID = 0
        testStation = self.globalData['stations'][testStationID]
        numBikesBefore = testStation.numBikes
        numRacksBefore = testStation.numRacks
        self.assertTrue(numRacksBefore > 0)

        # Two customers are waiting for bike pickup.
        testPickupQueue = self.globalData['pickupQueues'][testStationID]
        # One customer has waited too long and will leave the station. 
        longWaitCustomer = nycbike.Customer()
        longWaitCustomer.startID = testStationID
        longWaitCustomer.startPickupWait = currentTime - 100  # > REFUND_TIME
        testPickupQueue.put(longWaitCustomer)
        # The other customer has just arrived.
        shortWaitCustomer = nycbike.Customer()
        shortWaitCustomer.startID = testStationID
        shortWaitCustomer.startPickupWait = currentTime - 1  # < REFUND_TIME
        testPickupQueue.put(shortWaitCustomer)

        # The RideEnd event is scheduled and processed.
        customer = nycbike.Customer()
        customer.endID = testStationID
        rideEndEvent = engine.DiscreteEvent(
            nycbike.RideEnd, 10, globalData=self.globalData,
            stationID=testStationID, customer=customer)
        self.simEngine.schedule(rideEndEvent)
        self.simEngine.runSimulation(maxEvents=1)

        # The station attributes were updated.
        self.assertEqual(numBikesBefore + 1, testStation.numBikes)
        self.assertEqual(numRacksBefore - 1, testStation.numRacks)

        # Revenue is unchanged.
        self.assertEqual(0, self.globalData['statistics']['Revenue'])

        # No customers are waiting for pickup.
        self.assertEqual(0, len(testPickupQueue))
        # An Arrival event was scheduled for the short-wait customer.
        self.assertEqual(1, len(self.simEngine.FEL))
        self.assertEqual(nycbike.Arrival, self.simEngine.FEL[0].handler)
        self.assertEqual(
            shortWaitCustomer, self.simEngine.FEL[0].handlerKwargs['customer'])

    def test_rideEndEvent_noRacksAvailable(self):
        """Tests the RideEnd event when no racks are available."""
        currentTime = 100
        self.simEngine.simTime = currentTime
        self.globalData = self._initGlobalData(
            self.TEST_NUM_STATIONS, initEntities=True)

        # The station has zero racks available.
        testStationID = 0
        testStation = self.globalData['stations'][testStationID]
        numBikesBefore = testStation.numBikes
        testStation.numRacks = 0

        # One customer is waiting for bike pickup.
        testPickupQueue = self.globalData['pickupQueues'][testStationID]
        waitingCustomer = nycbike.Customer()
        waitingCustomer.startID = testStationID
        waitingCustomer.startPickupWait = currentTime - 1  # < REFUND_TIME
        testPickupQueue.put(waitingCustomer)

        # The RideEnd event is scheduled and processed.
        customer = nycbike.Customer()
        customer.endID = testStationID
        rideEndEvent = engine.DiscreteEvent(
            nycbike.RideEnd, 10, globalData=self.globalData,
            stationID=testStationID, customer=customer)
        self.simEngine.schedule(rideEndEvent)
        self.simEngine.runSimulation(maxEvents=1)

        # The station attributes were not changed.
        self.assertEqual(numBikesBefore, testStation.numBikes)
        self.assertEqual(0, testStation.numRacks)

        # Customers are still waiting for pickup.
        self.assertTrue(waitingCustomer in testPickupQueue)

        # The current customer entered the dropoff queue.
        self.assertTrue(
            customer in self.globalData['dropoffQueues'][testStationID])
        # The start of dropoff waiting time was recorded.
        self.assertEqual(rideEndEvent.timestamp, customer.startDropoffWait)


if __name__ == '__main__':
    unittest.main()
