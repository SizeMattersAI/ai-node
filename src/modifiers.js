function getLastModifier(modifiers) {
    // Sort modifiers by their numeric ID in descending order
    const sortedModifiers = [...modifiers].sort((a, b) => {
        const idA = parseInt(a.split('.')[0]);
        const idB = parseInt(b.split('.')[0]);
        return idB - idA;
    });

    // Get the first item (highest number) after sorting
    return sortedModifiers[0];
} 