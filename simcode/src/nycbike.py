"""NYC Citi bike sharing simulation."""

# Standard libs.
import argparse
import logging
import os
import time
import zipfile
from collections import deque

# Third-party libs.
import numpy as np

# App libs.
import simcode.src.data.trip_statistics.load_trip_stats as load_trip_stats
import simcode.src.engine as engine


##############################
###  Simulation constants  ###
##############################

# Duration of simulation in minutes.
DAY_DURATION = 24 * 60

# Fee for renting a bike.
TRIP_COST = 20

# If a customer has to wait more than this amount of minutes, they will get a
# refund (if waiting for dropoff), or they will use different mode of transport
# (if waiting for pickup).
REFUND_TIME = 5

# Probability of a bike becoming lost/damaged due to accident.
BIKE_LOSS_PROBABILITY = 0.001

# Initial number of bikes available in the system.
NUM_BIKES = 12000

# Number of bike racks (total) at each station. This is the maximum parking
# capacity for every station.
RACKS = 30


############################
###  Entity definitions  ###
############################

class Customer(object):
    """Represents a Citi bike customer."""

    # Monotonically increasing customer id.
    currentCustomerID = 0

    def __init__(self):
        # Assign unique customer ID.
        self.customerID = Customer.currentCustomerID
        Customer.currentCustomerID += 1
        self.startPickupWait = None


class Queue(object):
    """Represents queue of customers waiting for bike pickup or return."""

    def __init__(self):
        self.queue = deque([])

    def __len__(self):
        return len(self.queue)

    def __contains__(self, customer):
        return customer in self.queue

    def put(self, customer):
        """Insert the customer into the line."""
        self.queue.append(customer)

    def remove(self):
        return self.queue.popleft()


class Station(object):
    """Represents a bike station."""

    def __init__(self, stationID, totalRacks, numBikes):
        assert numBikes <= totalRacks
        self.stationID = stationID
        self.numBikes = numBikes
        self.numRacks = totalRacks - numBikes
        self.lastEvent = 0


########################
###  Event handlers  ###
########################

def Initialize(simEngine, **kwargs):
    """Initializes bike stations and schedules arrivals."""
    initialDistribution = kwargs['initialDistribution']
    globalData = kwargs['globalData']
    arrivalTimes = globalData['arrivalTimes']
    numStations = len(arrivalTimes)

    # Initialize bike stations and queues.
    for stationID in range(numStations):
        globalData['stations'].append(
            Station(stationID,
                    kwargs['racksPerStation'], initialDistribution[stationID]))
        globalData['pickupQueues'].append(Queue())
        globalData['dropoffQueues'].append(Queue())

    # Schedule first arrival event for each station.
    for stationID in range(numStations):
        if len(arrivalTimes[stationID]) > 0:
            t = arrivalTimes[stationID].pop(0)
            simEngine.schedule(engine.DiscreteEvent(
                Arrival, t, stationID=stationID, globalData=globalData))

def endSim(simEngine):
    """Collects simulation statistics at the end of the simulation period(24 hrs)"""
    for stationID in range(numStations):
        globalData['statistics']['IdleTime'][stationID] += globalData['stations'][stationID].numBikes * (currentTime - globalData['stations'][stationID].lastEvent)


def Arrival(simEngine, **kwargs):
    """Customer arrives at the station to pick up a bike."""
    globalData = kwargs['globalData']
    stationID = kwargs['stationID']
    numStations = globalData['destinationP'].shape[0]
    numTimeframes = globalData['destinationP'].shape[1]
    # Customer who will pick up a bike.
    if 'customer' in kwargs:
        customer = kwargs['customer']
    else:
        customer = Customer()
        customer.startID = stationID
    currentTime = simEngine.currentTime()

    # Checks the ArrivalData for the next arrival and schedules it.
    # Note: schedule immediately in case customer has to wait in line / leaves
    if len(globalData['arrivalTimes'][stationID]) > 0:
        t = globalData['arrivalTimes'][stationID].pop(0)
        simEngine.schedule(engine.DiscreteEvent(
                Arrival, t, stationID=stationID, globalData=globalData))

    # Check if there are bikes available.
    if globalData['stations'][stationID].numBikes <= 0:
        # Customer begins waiting for bike to become available.
        customer.startPickupWait = currentTime
        globalData['pickupQueues'][stationID].put(customer)
        logging.debug(
            '\t(customer %d) Damn! where are all the bikes at station %d, the time is %.3f' % (
            customer.customerID, stationID, currentTime))
        return

    # Customer pays to rent bike.
    globalData['statistics']['Revenue'] += TRIP_COST

    # Select destination based on the probabilities.
    currentTimeframe = int(np.floor(
        (currentTime / float(DAY_DURATION)) * numTimeframes))
    currentTimeframe = min(currentTimeframe, numTimeframes - 1)
    customer.endID = np.random.choice(
        numStations,
        p=globalData['destinationP'][stationID][currentTimeframe])

    # Schedule end of ride using the average trip duration.
    t = (currentTime
        + globalData['tripDurations'][stationID][customer.endID])

    # Determine if bike will become lost or damaged.
    rideOutcome = RideEnd
    if np.random.random() <= globalData['bikeLossProb']:
        rideOutcome = RideCrash
    simEngine.schedule(engine.DiscreteEvent(rideOutcome, t,
            customer=customer, globalData=globalData))

    # Update total Idle Time till current time
    if currentTime <= 1440:
        globalData['statistics']['IdleTime'][stationID] += globalData['stations'][stationID].numBikes * (currentTime - globalData['stations'][stationID].lastEvent)
        globalData['stations'][stationID].lastEvent = currentTime

    # Update number of bikes and racks for the station.
    globalData['stations'][stationID].numBikes -= 1
    globalData['stations'][stationID].numRacks += 1

    logging.debug(
        '\t(customer %d) yay! i got a bike from %d at time %.3f n im going to %d n will reach at %.3f' % (
        customer.customerID, stationID, currentTime, customer.endID, t))

    # Checks if there are people waiting to put the bikes back.
    if len(globalData['dropoffQueues'][stationID]) > 0:
        # Calculate time the customer waited to drop off the bike.
        waitingCustomer = globalData['dropoffQueues'][stationID].remove()
        waitTime = currentTime - waitingCustomer.startDropoffWait
        #  Update total wait time.
        globalData['statistics']['TimeWaitForDropoff'][stationID] += waitTime
        logging.debug(
            '\t(customer %d) finally i can return my bike at stn %d after waiting for %.3f having arrived at %.3f' % (
            waitingCustomer.customerID, stationID, waitTime, currentTime))
        # If customer has waited too long to return the bike, refund is given.
        if waitTime > REFUND_TIME:
            globalData['statistics']['Revenue'] -= TRIP_COST
            logging.debug(
                '\t(customer %d) at least i got my refund for waiting too long to return the bike' % (
                waitingCustomer.customerID))
        # Schedule RideEnd for the waiting customer.
        simEngine.schedule(engine.DiscreteEvent(
            RideEnd, currentTime,
            customer=waitingCustomer, globalData=globalData))


def RideEnd(simEngine, **kwargs):
    """Customer finishes the bike ride."""

    globalData = kwargs['globalData']
    customer = kwargs['customer']
    stationID = customer.endID
    currentTime = simEngine.currentTime()

    # Check if there are empty racks to keep the bike.
    if globalData['stations'][stationID].numRacks <= 0:
        # No empty racks. The customer begins waiting in queue.
        customer.startDropoffWait = currentTime
        globalData['dropoffQueues'][stationID].put(customer)
        logging.debug(
            '\t(customer %d) damn there are no empty racks at station %d at time %.3f' % (
            customer.customerID, stationID, currentTime))
        return
    # Update total Idle Time till current time
    if currentTime < 1440:
        globalData['statistics']['IdleTime'][stationID] += globalData['stations'][stationID].numBikes * (currentTime - globalData['stations'][stationID].lastEvent)
        globalData['stations'][stationID].lastEvent = currentTime

    # Customer returns the bike to the rack.
    globalData['stations'][stationID].numRacks -= 1
    globalData['stations'][stationID].numBikes += 1
    logging.debug(
        '\t(customer %d) perfecto! i reached my destination %d at time %.3f, my journey is complete' % (
        customer.customerID, stationID, currentTime))

    # If there is at least one customer waiting for a bike and waittime < 5
    # mins, schedule arrival event. Note: not every customer waiting for a
    # bike eventually takes a bike..customers leave after 5 mins.
    while(True):
        # Check if customers are waiting.
        if len(globalData['pickupQueues'][stationID]) == 0:
            break

        waitingCustomer = globalData['pickupQueues'][stationID].remove()
        waitTime = currentTime - waitingCustomer.startPickupWait
        globalData['statistics']['TimeWaitForCycle'][stationID] += waitTime
        if waitTime < REFUND_TIME:
            # Next waiting customer gets a bike
            simEngine.schedule(engine.DiscreteEvent(
                Arrival, currentTime,
                customer=waitingCustomer, stationID=stationID,
                globalData=globalData))
            logging.debug(
                '\t(customer %d) finally i get my ride at stn %d after waiting for %.3f having arrived at %.3f' % (
                waitingCustomer.customerID, stationID, waitTime, currentTime))
            break
        else:
            # We lose a customer
            globalData['statistics']['CustomersLost'][stationID] += 1
            logging.debug(
                '\t(customer %d) @#$%%! u wasted my time! i waited for %.3f minutes for a bike at stn %d, i dont want it anymore' % (
                waitingCustomer.customerID, waitTime, stationID))


def RideCrash(simEngine, **kwargs):
    """Bike is lost or damaged due to an accident."""
    globalData = kwargs['globalData']
    customer = kwargs['customer']

    # The bicycle is not returned to the station.
    globalData['statistics']['BikesLost'] += 1
    logging.debug(
            '\t(customer %d) oops! the bike was lost or damaged and I never reached stn %d' % (
            customer.customerID, customer.endID))


###########################
###  Simulation runner  ###
###########################

class BikeSharingSimulation(object):
    """Initializes and runs the bike sharing simulation."""

    def computeArrivalTimes(self, tripCountData):
        """Computes simulation arrival times based on the trip count data."""
        totalArrivalEvents = 0
        arrivalTimes = []
        numTimeframes = tripCountData.shape[1]
        timeframeLength = float(DAY_DURATION) / numTimeframes
        for stationID, arrivals in enumerate(tripCountData):
            arrivalTimes.append([])
            for i, numPeople in enumerate(arrivals):
                # Schedule events only if people arrived in this timeframe.
                totalArrivalEvents += numPeople
                if numPeople == 0:
                    continue
                firstArrival = i * timeframeLength
                interArrivalTime = timeframeLength / float(numPeople)
                for j in range(numPeople):
                    t = firstArrival + j * interArrivalTime
                    arrivalTimes[stationID].append(t)
        logging.info('Total Arrival events: %d' % totalArrivalEvents)
        return arrivalTimes

    def almostUniformWithTotalSum(self, d, totalSum):
        """Computes uniform or almost-uniform distribution.

        Args:
            d: Integer dimension of the distribution.
            totalSum: Integer total sum of the integers in the
                (almost-)uniform distribution.

        Returns:
            Uniform or almost-uniform (elements differ by at most 1)
            distribution having specified length and total sum.
        """
        m = int(totalSum / float(d))
        distr = np.ones(d) * m
        leftover = totalSum - m * d

        if leftover > 0:
            distr[:leftover] += 1

        assert sum(distr) == totalSum
        return distr

    def run(self, initialDistribution=None,
            totalNumBikes=NUM_BIKES, racksPerStation=RACKS, scaleArrivalRate=1,
            rngSeed=None, tripDataDir=None):
        """Runs the store checkout simulation until it completes.

        Args:
            initialDistribution: Initial distribution of bikes to stations.
            totalNumBikes: Total number of bikes used in the simulation. This
                parameter is ignored if initialDistribution is specified.
            racksPerStation: Number of bike racks per station.
            scaleArrivalRate: Scale factor for number of arrivals that occur
                during the simulation.

        Returns:
            Dictionary of simulation results.
        """
        simStartTime = time.time()
        logging.info('Citi Bike Sharing Simulation')
        logging.info('\ttotalNumBikes: %d' % totalNumBikes)
        logging.info('\tracksPerStation: %d' % racksPerStation)
        logging.info('\tscaleArrivalRate: %.3f' % scaleArrivalRate)

        # Seed RNG if specified.
        if rngSeed is not None:
            logging.info('RNG seed: %d' % rngSeed)
            np.random.seed(rngSeed)

        # Load statistics derived from the Citi Bike trip dataset.
        tripDataDir = {'tripDataDir': tripDataDir} if tripDataDir else {}
        tripCountData, tripDurations, destinationP = (
            load_trip_stats.loadTripStatistics(**tripDataDir))
        numStations = tripCountData.shape[0]

        # Compute arrival times based on trip count data for each station and
        # the arrival rate scale factor.
        tripCountData = np.rint(tripCountData * scaleArrivalRate).astype(int)
        arrivalTimes = self.computeArrivalTimes(tripCountData)

        # Initial distribution of bikes to stations (set at time 00:00).
        if initialDistribution is None:
            initialDistribution = self.almostUniformWithTotalSum(
                numStations, totalNumBikes)
        assert len(initialDistribution) == len(tripCountData)

        # Initialize simulation statistics.
        statistics = {
            'Revenue': 0,
            'TimeWaitForDropoff': np.zeros(numStations),
            'TimeWaitForCycle': np.zeros(numStations),
            'CustomersLost': np.zeros(numStations),
            'BikesLost': 0,
            'IdleTime': np.zeros(numStations),
        }

        # Global simulation variables.
        globalData = {
            # Entities.
            'stations': [],
            'pickupQueues': [],
            'dropoffQueues': [],
            # Citi bike dataset statistics.
            'arrivalTimes': arrivalTimes,
            'tripDurations': tripDurations,
            'destinationP': destinationP,
            # Simulation statistics.
            'statistics': statistics,
            # Constants.
            'bikeLossProb': BIKE_LOSS_PROBABILITY,
        }

        # Initialize the simulation engine.
        simEngine = engine.DiscreteEventSimulationEngine()

        # Schedule initial event.
        initEvent = engine.DiscreteEvent(
            Initialize, -1, globalData=globalData,
            initialDistribution=initialDistribution,
            racksPerStation=racksPerStation)
        simEngine.schedule(initEvent)
        endEvent = engine.DiscreteEvent(endSim, 1440)
        # Run the simulation.
        simEngine.runSimulation()
        simDuration = time.time() - simStartTime
        logging.info('Simulation complete. Took %.3f seconds.\n' % simDuration)

        # Report simulation statistics.
        logging.info('Revenue: %.2f dollars' % statistics['Revenue'])
        logging.info(
            'TimeWaitForCycle: %.3f minutes'
            % statistics['TimeWaitForCycle'].sum())
        logging.info(
            'TimeWaitForDropoff: %.3f minutes'
            % statistics['TimeWaitForDropoff'].sum())
        logging.info('CustomersLost: %d' % statistics['CustomersLost'].sum())
        logging.info('BikesLost: %d' % statistics['BikesLost'])
        logging.info('TotalIdleTime: %d' % statistics['IdleTime'].sum())

        return statistics


def main():
    """Parses command-lines args and runs the simulation."""
    parser = argparse.ArgumentParser(description='Bike Sharing Simulation')
    # Logging parameters.
    parser.add_argument('--loglevel', dest='loglevel', action='store',
        default='WARNING', help='Level of logging output.')
    parser.add_argument('--logfile', dest='logfile', action='store',
        default=None, help='Filename for logging output.')
    # Parameters used in experiments.
    parser.add_argument('--totalNumBikes', dest='totalNumBikes',
        action='store', default=NUM_BIKES, help='Total number of bikes.')
    parser.add_argument('--racksPerStation', dest='racksPerStation',
        action='store', default=RACKS, help='Number of racks per station.')
    parser.add_argument('--scaleArrivalRate', dest='scaleArrivalRate',
        action='store', default=1, help='Scale factor for arrival rate.')

    args = parser.parse_args()

    # Set the logging level.
    logging.basicConfig(
        filename=args.logfile, level=getattr(logging, args.loglevel.upper()))

    # Run the simulation.
    BikeSharingSimulation().run(
        totalNumBikes=int(args.totalNumBikes),
        racksPerStation=int(args.racksPerStation),
        scaleArrivalRate=float(args.scaleArrivalRate))


if __name__ == '__main__':
    main()
