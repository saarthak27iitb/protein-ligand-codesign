"""Fixed-sequence, restrained implicit-solvent MD for local robustness pilots.

Short trajectories are local perturbations, not an equilibrated population or a
model of all biologically relevant states. No mutations or coordinate noise.
"""
import json
from pathlib import Path

def make_conformations(protein,output,seed=42,frames=3,steps_per_frame=1000,platform_name=None,minimization_steps=100):
    from pdbfixer import PDBFixer
    from openmm import app,unit,Platform,LangevinMiddleIntegrator,CustomExternalForce
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    protein=Path(protein)
    clean=output/'protein.pdb'
    clean.write_text('\n'.join(x for x in protein.read_text().splitlines() if x.startswith(('ATOM  ','TER')))+'\nEND\n')
    print('Preparing fixed-sequence protein',flush=True)
    fixer=PDBFixer(filename=str(clean))
    fixer.missingResidues={}
    fixer.findMissingAtoms(); fixer.addMissingAtoms(seed=seed)
    ff=app.ForceField('amber14-all.xml','implicit/obc2.xml')
    modeller=app.Modeller(fixer.topology,fixer.positions)
    modeller.addHydrogens(ff,pH=7.,platform=Platform.getPlatformByName(platform_name or 'CPU'))
    print('Hydrogens added',flush=True)
    sequence=[(r.chain.id,r.id,r.name) for r in modeller.topology.residues()]
    system=ff.createSystem(modeller.topology,nonbondedMethod=app.NoCutoff,constraints=app.HBonds)
    restraint=CustomExternalForce('0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.addGlobalParameter('k',418.4)
    for parameter in ['x0','y0','z0']:
        restraint.addPerParticleParameter(parameter)
    for atom in modeller.topology.atoms():
        if atom.name in ['N','CA','C','O']:
            restraint.addParticle(atom.index,modeller.positions[atom.index].value_in_unit(unit.nanometer))
    system.addForce(restraint)
    integrator=LangevinMiddleIntegrator(300*unit.kelvin,1/unit.picosecond,0.002*unit.picoseconds)
    integrator.setRandomNumberSeed(seed)
    platforms=[Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
    name=platform_name or ('CUDA' if 'CUDA' in platforms else 'CPU')
    platform=Platform.getPlatformByName(name)
    props={'Precision':'mixed'} if name=='CUDA' else {'Threads':'4'}
    simulation=app.Simulation(modeller.topology,system,integrator,platform,props)
    simulation.context.setPositions(modeller.positions)
    print('Minimizing on '+name,flush=True)
    simulation.minimizeEnergy(maxIterations=minimization_steps)
    print('Integrating local conformations',flush=True)
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin,seed)
    files=[]
    for i in range(frames):
        simulation.step(steps_per_frame)
        state=simulation.context.getState(getPositions=True,getEnergy=True)
        path=output/f'frame_{i}.pdb'
        with path.open('w') as f:
            app.PDBFile.writeFile(modeller.topology,state.getPositions(),f,keepIds=True)
        files.append(str(path.resolve()))
    manifest={'files':files,'sequence':sequence,'sequence_fixed':True,'seed':seed,'platform':name,
              'temperature_K':300,'steps_per_frame':steps_per_frame,'timestep_ps':0.002,
              'minimization_max_iterations':minimization_steps,
              'backbone_restraint_kJ_mol_nm2':418.4,'force_field':['amber14-all.xml','implicit/obc2.xml'],
              'interpretation':'short restrained local perturbations; not equilibrium state populations'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    del simulation,integrator
    return manifest
