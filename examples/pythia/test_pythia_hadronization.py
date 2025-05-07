# two_gluon_gun.py: A two-gluon particle gun using Pythia8 with Python

import pythia8

# Initialize Pythia
pythia = pythia8.Pythia()

# Switch off process level but enable parton level
pythia.readString("ProcessLevel:all = off")
pythia.readString("PartonLevel:all = on")
pythia.readString("Check:event = on")

pythia.readString("Next:numberShowInfo = 0")
pythia.readString("Next:numberShowProcess = 0")
pythia.readString("Next:numberShowEvent = 0")

# Enable parton shower and hadronization
pythia.readString("PartonLevel:FSR = on")
pythia.readString("PartonLevel:ISR = on")
pythia.readString("HadronLevel:all = on")

# Set particle energy and number of events
energy = 20.0  # GeV
n_events = 10000

# Initialize Pythia
pythia.init()

for i_event in range(n_events):
    # Reset event record for new event
    pythia.event.reset()

    # Append two gluons with opposite momenta and color connection
    col1, col2 = 101, 102
	# pythia.event.append(21, 23, 0, 0, col1, col2, 0, 0, energy, 0, 0, energy)
	# pythia.event.append(21, 23, 0, 0, col2, col1, 0, 0, -energy, 0, 0, energy)
    pythia.event.append( 21, 23, 101, 102, 0., 0.,  energy, energy)
    pythia.event.append( 21, 23, 102, 101, 0., 0., -energy, energy)

    # pythia.event.append( 2212, -12, 0, 0, 3, 5, 0, 0, 0., 0., energy, energy, 0.)
    # pythia.event.append(-2212, -12, 0, 0, 6, 8, 0, 0, 0., 0., energy, energy, 0.)

    pythia.event[1].scale(energy)
    pythia.event[2].scale(energy)
    pythia.forceTimeShower(1, 2, energy)

    # Generate event and print first few events
    # if not pythia.next():
    #    print("Error generating event!")
    #    break
    if i_event < 5:
        pythia.event.list()

    # Print final particle information after hadronization
    print(f"\nEvent {i_event + 1} - Final Particles:")
    for particle in pythia.event:
        if particle.isFinal():
            print(f"ID: {particle.id()} Name: {pythia8.ParticleData().name(particle.id())} pT: {particle.pT():.2f} eta: {particle.eta():.2f} phi: {particle.phi():.2f}")


# Print statistics
pythia.stat()

print("Two-gluon particle gun simulation with parton shower and hadronization complete.")

