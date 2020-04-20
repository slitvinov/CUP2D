/* main_RL_cStart.cpp
 * Created by Ioannis Mandralis (ioannima@ethz.ch)
 * Main script for a single CStartFish learning a fast start
*/

#include <unistd.h>
#include <sys/stat.h>
#include "smarties.h"
#include "Simulation.h"
#include "Obstacles/CStartFish.h"


using namespace cubism;

// Task based smarties application.
// A task is something that shares the same agent, state, action-set, terminal condition, and initial condition.
class Task
{
public:
    const unsigned maxLearnStepPerSim = 200;
    // Simulation quantities needed for reward functions
    double timeElapsed = 0.0; // time elapsed until now in the episode
    double energyExpended = 0.0; // energy expended until now in the episode
public:
    inline void setAction(CStartFish* const agent,
                          const std::vector<double> act, const double t)
    {
        agent->act(t, act);
    }

    inline bool checkNaN(std::vector<double>& state, double& reward)
    {
        bool bTrouble = false;
        if(std::isnan(reward)) bTrouble = true;
        for(size_t i=0; i<state.size(); i++) if(std::isnan(state[i])) bTrouble = true;
        if ( bTrouble ) {
            reward = -100;
            printf("Caught a nan!\n");
            state = std::vector<double>(state.size(), 0);
        }
        return bTrouble;
    }

    inline void setTimeElapsed(const double& t) {
        timeElapsed = t;
    }

    inline void setEnergyExpended(const double& e) {
        energyExpended = e;
    }

};

// Target following
class GoToTarget : public Task
{
public:
    // Task constants
    std::vector<double> lower_action_bound{-4, -1, -1, -6, -3, -1.5, 0, 0};
    std::vector<double> upper_action_bound{+4, +1, +1, 0, 0, 0, +1, +1};
    const int nActions = 8;
    const int nStates = 15;
public:
    inline void resetIC(CStartFish* const a, smarties::Communicator*const c)
    {
        double initialAngle = -98.0;
        double length = a->length;
        double com[2] = {0.5, 0.5};
        double target[2] = {0.0, 0.0};


        // Place target at 1.5 fish lengths away (1.5 * 0.25 = 0.375).
        double targetRadius_ = 1.5;
        double targetRadius = targetRadius_ * length;
//    double targetAngle = 180.0 + initialAngle; // Diametrically opposite from initial orientation.
        double targetAngle = initialAngle; // Directly ahead of fish
        target[0] = targetRadius * std::cos(targetAngle);
        target[1] = targetRadius * std::sin(targetAngle);

        // Set agent quantities
        a->setCenterOfMass(com);
        a->setOrientation(initialAngle * M_PI / 180.0);
        a->setTarget(target);
    }

    inline bool isTerminal(const CStartFish*const a) {
        // Terminate after time equal to Gazzola's C-start
        printf("Time of current episode is: %f\n", timeElapsed);
        return timeElapsed > 1.5882352941 ;
    }

    inline std::vector<double> getState(const CStartFish* const a)
    {
        return a->stateTarget();
    }

    inline double getReward(const CStartFish* const a) {
//    // Reward inspired from Zermelo's problem (without the penalty per time step)
//    // Current position and previous position relative to target in absolute coordinates.
//    double relativeX = getState(a)[0] * a->length;
//    double relativeY = getState(a)[1] * a->length;
//    double prevRelativeX = previousRelativePosition[0] * a->length;
//    double prevRelativeY = previousRelativePosition[1] * a->length;
//    // Current distance and previous distance from target in absolute units.
//    double distance_ = (std::sqrt(std::pow(relativeX, 2) + std::pow(relativeY, 2))) / a->length;
//    double prevDistance_ = (std::sqrt(std::pow(prevRelativeX, 2) + std::pow(prevRelativeY, 2))) / a->length;
//    double reward = 1/distance_ - 1/prevDistance_;
//    // Simpler reward structure
//    double radialDisplacement_ = a->getRadialDisplacement() / a->length;


        double distToTarget_ = getState(a)[0];
        double reward = -distToTarget_;

        printf("Stage reward is: %f \n", reward);
        return reward;
    }

};

// Escaping
class Escape : public Task
{
public:
    // Task constants
    std::vector<double> lower_action_bound{-4, -1, -1, -6, -3, -1.5, 0, 0};
    std::vector<double> upper_action_bound{0, 0, 0, 0, 0, 0, +1, +1};
    const int nActions = 8;
    const int nStates = 14;
public:

    inline void resetIC(CStartFish* const a, smarties::Communicator*const c)
    {
        double com[2] = {0.5, 0.5};
        a->setCenterOfMass(com);
        a->setOrientation(-98.0 * M_PI / 180.0);
    }

    inline bool isTerminal(const CStartFish*const a)
    {
        // Terminate when the fish exits a radius of 1.5 characteristic lengths or time > 2.0
        return (a->getRadialDisplacement() >= 1.50 * a->length) || timeElapsed > 2.0 ;
    }

    inline std::vector<double> getState(const CStartFish* const a)
    {
        return a->stateEscape();
    }

};
class DistanceEscape : public Escape
{
public:
    inline double getReward(const CStartFish* const a)
    {
        // Dimensionless radial displacement:
        double dimensionlessRadialDisplacement = a->getRadialDisplacement() / a->length;
        // Reward is dimensionless radial displacement:
        double reward = dimensionlessRadialDisplacement;
        printf("Stage reward is: %f \n", reward);
        return reward;
    }
};
class DistanceTimeEscape : public Escape
{
public:
    inline double getReward(const CStartFish* const a)
    {
        // Dimensionless radial displacement:
        double dimensionlessRadialDisplacement = a->getRadialDisplacement() / a->length;
        // Dimensionless episode time:
        double dimensionlessTElapsed = timeElapsed / a->Tperiod;
        // Stage reward
        double stageReward = dimensionlessRadialDisplacement;
        // Terminal reward
        double terminalReward = dimensionlessRadialDisplacement - dimensionlessTElapsed;
        // Overall reward
        double reward = isTerminal(a)? terminalReward : stageReward;
        printf("Stage reward is: %f \n", reward);
        return reward;
    }
};
class DistanceTimeEnergyEscape : public Escape
{
public:
    inline double getReward(const CStartFish* const a)
    {
        // Baseline energy consumed by a C-start:
        const double baselineEnergy = 0.011; // in joules
        // Relative difference in energy of current policy with respect to normal C-start:
        double relativeChangeEnergy = std::abs(energyExpended - baselineEnergy) / baselineEnergy;
        // Dimensionless radial displacement:
        double dimensionlessRadialDisplacement = a->getRadialDisplacement() / a->length;
        // Dimensionless episode time:
        double dimensionlessTElapsed = timeElapsed / a->Tperiod;

        // Stage reward
        double stageReward = dimensionlessRadialDisplacement;
        // Terminal reward
        double terminalReward = dimensionlessRadialDisplacement - relativeChangeEnergy - dimensionlessTElapsed;
        // Overall reward
        double reward = isTerminal(a)? terminalReward : stageReward;

        printf("Stage reward is: %f \n", reward);
        return reward;
    }
};

inline void app_main(
        smarties::Communicator*const comm, // communicator with smarties
        MPI_Comm mpicom,                   // mpi_comm that mpi-based apps can use
        int argc, char**argv               // args read from app's runtime settings file
) {
    // Get the task definition
    GoToTarget task = GoToTarget();

    // Define the maximum learn steps per simulation (episode)
    const unsigned maxLearnStepPerSim = task.maxLearnStepPerSim;

    for(int i=0; i<argc; i++) {printf("arg: %s\n",argv[i]); fflush(0);}
    comm->setStateActionDims(task.nStates, task.nActions);

    Simulation sim(argc, argv);
    sim.init();

    CStartFish*const agent = dynamic_cast<CStartFish*>( sim.getShapes()[0] );
    if(agent==nullptr) { printf("Agent was not a CStartFish!\n"); abort(); }

    comm->setActionScales(task.upper_action_bound, task.lower_action_bound, true);

    if(comm->isTraining() == false) {
        sim.sim.verbose = true; sim.sim.muteAll = false;
        sim.sim.dumpTime = agent->Tperiod / 20;
    }

    unsigned int sim_id = 0, tot_steps = 0;

    // Terminate loop if reached max number of time steps. Never terminate if 0
    while( true ) // train loop
    {
        if(comm->isTraining() == false)
        {
            char dirname[1024]; dirname[1023] = '\0';
            sprintf(dirname, "run_%08u/", sim_id);
            printf("Starting a new sim in directory %s\n", dirname);
            mkdir(dirname, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            chdir(dirname);
        }

        sim.reset();
        task.resetIC(agent, comm); // randomize initial conditions
        double target[2] = {0,0}; agent->getTarget(target);
        printf("Target is: (%f, %f)\n", target[0], target[1]);

        double t = 0, tNextAct = 0;
        unsigned int step = 0;
        bool agentOver = false;
        double energyExpended = 0.0; // Energy consumed by fish in one episode

        comm->sendInitState( task.getState(agent) ); //send initial state
        while (true) //simulation loop
        {
            task.setAction(agent, comm->recvAction(), tNextAct);
            tNextAct = agent->getTimeNextAct();

            while (t < tNextAct)
            {
                // Get the time-step from the simulation
                const double dt = sim.calcMaxTimestep();
                t += dt;

                // Set the task-time
                task.setTimeElapsed(t);
                printf("Time is: %f\n", t);

                // Forward integrate the energy expenditure
                energyExpended += -agent->defPowerBnd * dt; // We want work done by fish on fluid.

                // Set the task-energy-expenditure
                task.setEnergyExpended(energyExpended);
                printf("Energy is: %f\n", energyExpended);

                if ( sim.advance( dt ) ) { // if true sim has ended
                    printf("Set -tend 0. This file decides the length of train sim.\n");
                    assert(false); fflush(0); abort();
                }

                if ( task.isTerminal(agent)) {
                    agentOver = true;
                    break;
                }
            }
            step++;
            tot_steps++;
            std::vector<double> state = task.getState(agent);
            double reward = task.getReward(agent);
            printf("Reward is: %f\n", reward);

            if (agentOver || task.checkNaN(state, reward)) {
                printf("Agent failed\n"); fflush(0);
                comm->sendTermState(state, reward);
                break;
            }
            else
            if (step >= maxLearnStepPerSim) {
                printf("Sim ended\n"); fflush(0);
                comm->sendLastState(state, reward);
                break;
            }
            else comm->sendState(state, reward);
        } // simulation is done

        if(comm->isTraining() == false) chdir("../");
        sim_id++;

        if (comm->terminateTraining()) return; // exit program
    }

}

int main(int argc, char**argv)
{
    smarties::Engine e(argc, argv);
    if( e.parse() ) return 1;
    e.run( app_main );
    return 0;
}
