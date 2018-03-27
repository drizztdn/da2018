import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.show()

plt.plot([1,2,3,4], [1,4,9,16], 'mo-')
plt.axis([0, 6, 0, 20])
plt.show()

