from enum import Enum

#Hard coded and copied from generatorTradeAndNonTrade.py for now 
#(obv would be good to clean this up)
itemList = ["GOLD", "STONES", "WOOD", "WHEAT", "COAL", "IRON", "ALUMINUM", "HORSES"]
Resources = Enum('Resources', itemList)

#Note - probably a bad way of doing this but just quick prototyping
class ResourceStats:
    def __init__(self, count, value):
        self.count = count
        self.value = value

    def UpdateCount(self, delta):
        self.count += delta

        if self.count < 0:
            self.count = 0

    def GetCount(self):
        return self.count

    def IsEmpty(self):
        return self.count <= 0

    def GetValue(self):
        return self.value

    def __str__(self):
        return "\tCount: " + str(self.count) + "\n\t" + "Value: " + str(self.value)

    

class Player:
    def __init__(self):
        #Dict of resource enum to ResourceStats obj
        self.resources = {}
        #Set of resource names for quick check
        self.resourceNames = {}
        #Map to access values of resources
        self.resourceValMap = {}

    #Resources expected to have key=resource enum name, val=amt
    def SetResources(self, resourcesDict):
        self.resources = resourcesDict

        for resource in self.resources.keys():
            val = self.resources[resource].GetValue()
            self.resourceNames[resource.name.lower()] = resource
            self.resourceValMap[resource] = val

    def HasResource(self, resource):
        if resource in self.resourceNames:
            if self.resourceNames[resource] in self.resources:
                return True, self.resources[self.resourceNames[resource]]

        return False, -1

    def SetResourceValueMap(self, resource, value):
        self.resourceValMap[resource] = value
        self.resourceNames[resource.name.lower()] = resource

    def GetResourceValue(self, resource):
        if resource in self.resourceNames.keys():
            return self.resourceValMap[self.resourceNames[resource]]

        return None

    def UpdateResourceCount(self, resource, delta):
        resourceEnum = None

        if resource in self.resourceNames.keys():
            resourceEnum = self.resourceNames[resource]

        if resourceEnum != None:
            #Add resource to inventory if not there before
            if resourceEnum not in self.resources.keys():
                self.resources[resourceEnum] = ResourceStats(delta, self.resourceValMap[resourceEnum])
            #Otherwise, update inventory count
            else:
                self.resources[resourceEnum].UpdateCount(delta)
        else:
            print("Game Log --- Error finding resource to update")

    def DebugPrintResources(self, showValue=True):
        for resource in self.resources.keys():
            print("Resource: " + resource.name)
            if showValue:
                print(self.resources[resource])
            else:
                print("\tCount: " +  str(self.resources[resource].GetCount()))

    def DebugPrintResourceValueMap(self):
        for resource in self.resourceValMap.keys():
            print("Resource: " + resource.name, end=", ")
            print("Value: " + str(self.resourceValMap[resource]))
    

'''test = Player()
test.SetResources({Resources.GOLD : ResourceStats(5,1), Resources.STONES : ResourceStats(3,2)})
test.SetResourceValueMap(Resources.HORSES, 3)
test.UpdateResourceCount("horses", 2)
test.UpdateResourceCount("stones", 2)
test.DebugPrintResources()'''