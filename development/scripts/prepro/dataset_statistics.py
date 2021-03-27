
import os, sys, yaml, pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


surfaces = ['Plane','Revolution', 'Cylinder','Extrusion','Cone','Other','Sphere','Torus','BSpline']

def parseYamlFile(path):
    data = None
    with open(path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data


def computeStats(basePath,limit):
    stats = {"faces": [], "edges": [], "verts": [], "surfs": [], "surfaceTypes": dict(zip(surfaces, [0]*len(surfaces)))}

    for root, _, fnames in sorted(os.walk(basePath)):
        for fname in fnames:
            if (os.path.splitext(fname)[1] == ".yml"):
                print(fname)
                data = parseYamlFile(os.path.join(root, fname))
                faces = data['#faces']
                stats["faces"].append(faces)
                verts = data['#verts']
                stats["verts"].append(verts)
                surfs = data['#surfs']
                stats["surfs"].append(surfs)

                surfTypes = stats["surfaceTypes"]
                for surf in data['surfs']:
                    if surf not in surfTypes:
                        print("Invalid surface",surf)
                    else:
                        surfTypes[surf] += 1
                if (limit != None) and (len(stats['faces'])>= limit):
                    return stats
    return stats

def plotSurfacePieChart(stats):
    plt.figure(figsize=(10, 10))
    plt.pie(stats["surfaceTypes"].values(), labels=stats["surfaceTypes"].keys(),autopct='%1.1f%%')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Surface types')

def plotFacesHistogram(stats):
    fig = plt.figure(figsize=(30, 10))
    plt.hist(stats['faces'], 1000,cumulative=True,histtype='step')
    plt.xscale('log')
    plt.xticks([1e3,5e3,1e4,2.5e4,5e4,1e5,1e6])
    fig.get_axes()[0].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    plt.grid()
    plt.title('Number of faces')

def plotSurfacesHistogram(stats):
    fig = plt.figure(figsize=(30, 10))
    plt.hist(stats['surfs'], 1000, cumulative=True, histtype='step')
    plt.xscale('log')
    plt.xticks([1, 10, 20, 50, 100, 200, 500,800])
    fig.get_axes()[0].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    plt.grid()
    #fig.get_axes()[0].get_xaxis().set_minor_locator(ticker.MultipleLocator(10))
    plt.title('Number of surfaces')

if len(sys.argv) < 2:
    print("Wrong parameters")
    exit(1)
elif len(sys.argv) < 3:
    basePath = sys.argv[-1]
    limit = None
else:
    basePath = sys.argv[-2]
    limit = int(sys.argv[-1])

print("Statistics for dataset in {}, limit {}".format(basePath,limit))

statPath = os.path.join(basePath,"stat")


statsFile = os.path.join(basePath, "total_stats_" + str(limit) +".pickle")
stats = {}
if (not os.path.exists(statsFile)):
    stats = computeStats(statPath,limit)
    pickle.dump(stats,open(statsFile,"wb"))
else:
    stats = pickle.load(open(statsFile,'rb'))


plotSurfacePieChart(stats)
plotFacesHistogram(stats)
plotSurfacesHistogram(stats)
plt.show()