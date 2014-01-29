var fs = require('fs');

var parseCSV = function(file, callback){
    fs.readFile(file, function(err, data){
        if(err){
            console.error(file + ": file not exist.");
            process.exit(1);
        }
        var csv = new Array();
        var lines = String(data).split('\n');
        for(var i = 0; i < lines.length; i++){
            if(lines[i].length > 0){
                csv.push(lines[i].split(','));
            }
        }
        callback(csv);
    });
};

var culcAverage = function(csv){
    var sum = 0;
    for(var i = 0; i < csv.length; i++){
        sum += Number(csv[i][1]);
    }
    console.log(csv[0][0] + ',' + String((sum / csv.length).toFixed(6)));
};


/*
 * main
 */
parseCSV(process.argv[2], culcAverage);
