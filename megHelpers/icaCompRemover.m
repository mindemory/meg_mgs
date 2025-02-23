function bad_comps = icaCompRemover(subjID, run)

% Reject components
switch subjID
    case 1 % sub-01
        switch run
            case 1
                bad_comps = [1 2 21 28 50];
            case 2
                bad_comps = [1 2 22 36 46];
            case 3
                bad_comps = [1 3 22 32 68];
            case 4
                bad_comps = [1 2 3 22 30 64];
            case 5
                bad_comps = [1 2 3 22 28 70 76];
            case 6
                bad_comps = [1 2 21 33 77];
            case 7
                bad_comps = [1 2 15 32 79 122]; % unsure of 122
            case 8
                bad_comps = [1 2 4 22 26 68 95];
        end
    case 2 % sub-02
        switch run
            case 1
                bad_comps = [1 2 3 4 5 7 11 20];
            case 2
                bad_comps = [1 2 3 4 5 11 16 35];
            case 3
                bad_comps = [1 2 3 4 5 6 13 25 30];
            case 4
                bad_comps = [1 2 3 4 5 23 24 26];
            case 5
                bad_comps = [1 2 3 4 5 11 16 30];
            case 6
                bad_comps = [1 2 3 4 5 13 20 24];
            case 7
                bad_comps = [1 2 3 5 14 20];
            case 8
                bad_comps = [1 2 3 4 28 33];
        end
    case 3 % sub-03
        switch run
            case 1
                bad_comps = [1 2 3 4 6 14];
            case 2
                bad_comps = [1 2 3 9 18];
            case 3
                bad_comps = [1 2 6 15 25];
            case 4
                bad_comps = [1 2 3 4 8 10 11 28];
            case 5
                bad_comps = [1 2 8 14 ];
            case 6
                bad_comps = [1 2 7 9];
            case 7
                bad_comps = [1 2 5 6];
            case 8
                bad_comps = [1 2 10 16 21];
        end
    case 4 % sub-04
        switch run
            case 1
                bad_comps = [1 2 3 4 5 6 7];
            case 2
                bad_comps = [1 2 3 4 5 20];
            case 3
                bad_comps = [1 2 3 5 6];
            case 4
                bad_comps = [1 2 3 5 7 108 119 136];
            case 5
                bad_comps = [1 2 3 4 7];
            case 6
                bad_comps = [1 2 3 5 10];
            case 7
                bad_comps = [1 2 3 4 7 ];
            case 8
                bad_comps = [1 2 3 6];
            case 9
                bad_comps = [1 3 4 5 7 120 129 138];
            case 10
                bad_comps = [1 2 3 6];
        end
    case 5 % sub-05
        switch run
            case 1
                bad_comps = [1 2 3 5 15];
            case 2
                bad_comps = [1 2 4 7];
            case 3
                bad_comps = [1 2 3 5 11];
            case 4
                bad_comps = [1 2 3 5 9];
            case 5
                bad_comps = [1 2 4 11];
            case 6
                bad_comps = [1 2 5 13];
            case 7
                bad_comps = [1 2 3 6 11];
            case 8
                bad_comps = [1 2 6 9];
            case 9
                bad_comps = [1 2 4 17];
            case 10
                bad_comps = [1 2 7 14];
            case 11
                bad_comps = [1 2 5 9];
            case 12
                bad_comps = [1 2 4 5 14];
        end
    case 6 % sub-06
        switch run
            case 1
                bad_comps = [1 2 13 16 18 21];
            case 2
                bad_comps = [1 2 10 14 16 18];
            case 3
                bad_comps = [1 2 7 16 18 24];
            case 4
                bad_comps = [1 2 3 13 15 18];
            case 5
                bad_comps = [1 2 9 15 19 25];
            case 6
                bad_comps = [1 2 8 19 21 24];
            case 7
                bad_comps = [1 2 8 16 19 23];
            case 8
                bad_comps = [1 2 11 22 24 28];
            case 9
                bad_comps = [1 2 3 10 20 22 28];
            case 10
                bad_comps = [1 2 3 8 17 27 34];
            case 11
                bad_comps = [1 2 14 20 22 27];
            case 12
                bad_comps = [1 2 3 7 19 22 34];
        end
    case 7 % sub-07
        switch run
            case 1
                bad_comps = [1 2 3 4 7 13 20];
            case 2
                bad_comps = [1 2 3 4 5];
            case 3
                bad_comps = [1 2 3 4 5 12];
            case 4
                bad_comps = [1 2 3 4 15];
            case 5
                bad_comps = [1 2 4 6 7 28];
            case 6
                bad_comps = [1 2 3 4 6 16 17];
            case 7
                bad_comps = [1 2 3 4 5];
            case 8
                bad_comps = [1 2 3 4 7 8];
            case 9
                bad_comps = [1 2 3 5 7 24];
            case 10
                bad_comps = [1 2 3 4];
            case 11
                bad_comps = [1 2 3 4 5 7 11 ];
        end
    case 8 % sub-08
        switch run
            case 1
                bad_comps = [1 2 6 13 15 27];
            case 2
                bad_comps = [1 2 3 14 19 23];
            case 3
                bad_comps = [1 2 3 10 16 23];
            case 4
                bad_comps = [1 2 3 12 20 23];
            case 5
                bad_comps = [1 2 3 16 22 33];
            case 6
                bad_comps = [1 2 3 4 11 18 21];
            case 7
                bad_comps = [1 2 13 18 19];
            case 8
                bad_comps = [1 2 3 7 12 16];
            case 9
                bad_comps = [1 2 3 4 5 17 24];
        end
    case 9 % sub-09
        switch run
            case 1
                bad_comps = [1 2 7 11 14];
            case 2
                bad_comps = [1 2 7 8 9 12];
            case 3
                bad_comps = [1 2 3 4 5 10 14];
            case 4
                bad_comps = [1 2 3 11 12 13 32];
            case 5
                bad_comps = [1 2 3 4 12 18 35];
            case 6
                bad_comps = [1 2 3 4 13 16];
            case 7
                bad_comps = [1 2 11 17];
            case 8
                bad_comps = [1 2 3 4 5 11 15];
            case 9
                bad_comps = [1 2 3 4 5 11 13];
            case 10
                bad_comps = [1 2 3 12 17 36];
        end
    case 10 % sub-10
        switch run
            case 1
                bad_comps = [1 2 3 6 15 24 28];
            case 2
                bad_comps = []; % Don't even run this
            case 3
                bad_comps = [1 2 4 18 25];
            case 4
                bad_comps = [1 3 5 7 11 26];
            case 5
                bad_comps = [1 2 4 8 19];
            case 6
                bad_comps = [1 2 3 8 14 21];
            case 7
                bad_comps = [1 2 7 8 14 27];
            case 8
                bad_comps = [1 2 3 11 16 55];
            case 9
                bad_comps = [1 2 4 8 15];
            case 10
                bad_comps = [1 2 7 8 15 36];
        end
    case 11 % sub-11
        switch run
            case 1
                bad_comps = [1 6 9 15];
            case 2
                bad_comps = [1 6 7 9 17];
            case 3
                bad_comps = [];
            case 4
                bad_comps = [1 2 7 8 18];
            case 5
                bad_comps = [1 2 7 8 11 25];
            case 6
                bad_comps = [1 5 7 8 17];
            case 7
                bad_comps = [2 8 9 14];
            case 9
                bad_comps = [1 2 5 7 9 12 23];
            case 10
                bad_comps = [1 3 5 7 8 23];
        end
    case 12 % sub-12
        switch run
            case 1
                bad_comps = [1 4 9 19 31];
            case 2
                bad_comps = [1 6 9 11 35];
            case 3
                bad_comps = [1 5 9 11 17 35];
            case 4
                bad_comps = [1 7 10 15 31 39 95];
            case 5
                bad_comps = [1 3 8 12 32];
            case 6
                bad_comps = [1 8 15 48 77 116];
            case 7
                bad_comps = [1 2 9 15 43 103];
            case 8
                bad_comps = [1 2 8 11 58 101 116];
            case 9
                bad_comps = [1 6 12 21 49 121];
            case 10
                bad_comps = [1 6 12 42 84];
            case 11
                bad_comps = [1 2 6 13 41 70];
        end
    case 13 % sub-13
        switch run 
            case 1
                bad_comps = [1 2 3 12 33];
            case 2
                bad_comps = [1 2 4 6 16];
            case 3
                bad_comps = [1 2 3 16];
            case 4
                bad_comps = [1 2 3 8];
            case 5
                bad_comps = [1 2 3 12];
            case 6
                bad_comps = [1 2 3 5 15];
            case 7
                bad_comps = [1 2 3 4 16];
            case 8
                bad_comps = [1 2 3 7 22];
        end
    case 15 % sub-15
        switch run
            case 1
                bad_comps = [1 2 3 4 5 14];
            case 2
                bad_comps = [1 2 3 4 15];
            case 3
                bad_comps = [1 2 3 4 14];
            case 4
                bad_comps = [1 2 3 4 6 16];
            case 5
                bad_comps = [1 2 3 4 6 14];
            case 6
                bad_comps = [1 2 3 4 16];
            case 7
                bad_comps = [1 2 3 6 15];
            case 8
                bad_comps = [1 2 3 5 6 14];
        end
    case 16 % sub-16
        switch run
            case 1
                bad_comps = [1 2 3 4 6 20];
            case 2
                bad_comps = [1 2 3 5 7];
            case 3
                bad_comps = [1 2 3 4 5];
            case 4
                bad_comps = [];
            case 5
                bad_comps = [];
            case 6
                bad_comps = [];
            case 7
                bad_comps = [];
            case 8
                bad_comps = [];
        end
    case 17 % sub-17
        switch run
            case 1
                bad_comps = [1 2 3 4 9 34];
            case 2
                bad_comps = [1 2 5 18 35];
            case 3
                bad_comps = [1 2 3 4 5 11 20];
            case 4
                bad_comps = [1 2 3 6 14 18];
            case 5
                bad_comps = [1 2 4 6 45];
            case 6
                bad_comps = [1 2 3 6 7 15];
            case 7
                bad_comps = [1 2 3 5 9];
            case 8
                bad_comps = [1 2 5 13 52];
            case 9
                bad_comps = [1 2 3 11 23];
            case 10
                bad_comps = [1 2 3 4 7 22 45];
        end
    case 18 % sub-18
        switch run
            case 1
                bad_comps = [1 2 3 5 7 15];
            case 2
                bad_comps = [1 2 3 4 6 25];
            case 3
                bad_comps = [1 2 3 5 6 13];
            case 4
                bad_comps = [1 2 3 4 7];
            case 5
                bad_comps = [1 2 3 4 5 9 20];
            case 6
                bad_comps = [1 2 3 4 6];
            case 7
                bad_comps = [1 2 3 4 6];
            case 8
                bad_comps = [1 2 3 4 6];
        end
    case 19 % sub-19
        switch run
            case 1
                bad_comps = [1 2 5 11 14];
            case 2
                bad_comps = [1 2 3 6 12 44];
            case 3
                bad_comps = [1 2 5 10 26];
            case 4
                bad_comps = [1 2 5 13 35];
            case 5
                bad_comps = [1 2 3 4 9 14];
            case 6
                bad_comps = [1 2 4 13 24];
            case 7
                bad_comps = [1 2 3 4 13 20];
            case 8
                bad_comps = [1 2 3 4 10 13 16 18 28 29];
            case 9
                bad_comps = [];
        end
    case 20 % sub-20
        switch run
            case 1
                bad_comps = [1 2 3 7 31];
            case 2
                bad_comps = [1 4 5 6 15 26];
            case 3
                bad_comps = [1 2 3 6 21 37];
            case 4
                bad_comps = [1 2 4 8 14 23];
            case 5
                bad_comps = [1 2 3 5 8 14 20];
            case 6
                bad_comps = [1 2 5 8 9 15];
            case 7
                bad_comps = [1 2 4 8 17 18];
            case 8
                bad_comps = [1 2 6 9 16 17];
            case 9
                bad_comps = [1 2 4 8 12 18];
        end
    case 21 % sub-21
        switch run
            case 1
                bad_comps = [];
            case 2
                bad_comps = [];
            case 3
                bad_comps = [];
            case 4
                bad_comps = [];
            case 5
                bad_comps = [];
            case 6
                bad_comps = [];
            case 7
                bad_comps = [];
            case 8
                bad_comps = [];
        end
    case 22 % sub-22
        if run == 1
            bad_comps = [1 2 3 4 6 14];
        elseif run == 2
            bad_comps = [1 2 3 4 5 7 13];
        elseif run == 3
            bad_comps = [1 2 3 5 15];
        elseif run == 4
            bad_comps = [1 2 3 4 5 17];
        elseif run == 5
            bad_comps = [1 2 3 5 6 11];
        elseif run == 6
            bad_comps = [1 2 3 5 6 11];
        elseif run == 7
            bad_comps = [1 2 4 11 17];
        elseif run == 8
            bad_comps = [1 2 3 5 10 11];
        else
            bad_comps = [1 3 5 6 11 17];
        end
    case 23 % sub-23
        switch run
            case 1
                bad_comps = [1 2 5 12 14];
            case 2
                bad_comps = [1 2 3 16];
            case 3
                bad_comps = [1 2 3 6 18];
            case 4
                bad_comps = [1 2 13 18];
            case 5
                bad_comps = [1 2 3 16 18];
            case 6
                bad_comps = [1 2 3 7 15];
            case 7
                bad_comps = [1 2 3 16];
            case 8
                bad_comps = [1 2 4 17];
            case 9
                bad_comps = [1 2 4 17];
        end
    case 24 % sub-24
        switch run
            case 1
                bad_comps = [1 2 3 6 8 15];
            case 2
                bad_comps = [1 2 3 6 16];
            case 3
                bad_comps = [1 2 3 6 17 30];
            case 4
                bad_comps = [1 2 3 4 7 9 17];
            case 5
                bad_comps = [1 2 3 4 6 9 20];
            case 6
                bad_comps = [1 2 3 4 5 11 18];
            case 7
                bad_comps = [1 2 3 4 5 9 24];
            case 8
                bad_comps = [1 2 3 4 5 9 23];
        end
    case 25 % sub-25
        switch run
            case 1
                bad_comps = [1 3 8 14 28 38];
            case 2
                bad_comps = [1 3 15 28 37]; 
            case 3
                bad_comps = [1 2 4 16 31 35];
            case 4
                bad_comps = [1 3 17 23 39];
            case 5
                bad_comps = [1 3 21 27 43];
            case 6
                bad_comps = [1 2 3 16 30 54];
            case 7
                bad_comps = [1 2 5 20 33 43];
            case 8 % Not analyzed
                bad_comps = [];
        end
    case 26 % sub-26
        switch run
            case 1
                bad_comps = [1 2 3 6 15];
            case 2
                bad_comps = [1 2 3 4 5 13];
            case 3
                bad_comps = [1 2 3 4 5 22];
            case 4
                bad_comps = [1 2 3 4 5 14];
            case 5
                bad_comps = [1 2 3 4 5 19];
            case 6
                bad_comps = [1 2 3 4 5 6 14];
            case 7
                bad_comps = [1 2 3 4 11];
            case 8
                bad_comps = [1 2 3 4 5 27];
        end
    case 27 % sub-27
        switch run
            case 1
                bad_comps = [1 2 3 4 6 11];
            case 2
                bad_comps = [1 2 3 4 6 14];
            case 3
                bad_comps = [1 2 3 4 5 13];
            case 4
                bad_comps = [1 2 3 4 5 9];
            case 5
                bad_comps = [1 2 3 4 6 8];
            case 6
                bad_comps = [1 2 3 4 7 8];
            case 7
                bad_comps = [1 2 3 4 7 13];
            case 8
                bad_comps = [1 2 3 4 6 9];
        end
    case 28 % sub-28
        switch run
            case 1
                bad_comps = [1 2 3 4 7 11];
            case 2
                bad_comps = [1 2 3 5 7 8];
            case 3
                bad_comps = [1 2 3 4 13 15];
            case 4
                bad_comps = [1 2 12 13 24];
            case 5
                bad_comps = [1 2 13 14 15 ];
            case 6
                bad_comps = [1 2 3 7 13 15 ];
            case 7
                bad_comps = [1 2 3 8 9];
            case 8
                bad_comps = [1 2 3 9 10 15];
        end
    case 29 % sub-29
        switch run
            case 1
                bad_comps = [1 6 13 39];
            case 2
                bad_comps = [1 2 7 9 36];
            case 3
                bad_comps = [1 5 8 25];
            case 4
                bad_comps = [1 4 7 40];
            case 5
                bad_comps = [1 6 9 41];
            case 6
                bad_comps = [1 8 11 55];
            case 7
                bad_comps = [1 2 8 41];
            case 8
                bad_comps = [1 3 7 29];
        end
    case 30 % sub-30
        switch run
            case 1
                bad_comps = [1 2 4];
            case 2
                bad_comps = [1 2 3 27];
            case 3
                bad_comps = [1 2 4 20];
            case 4
                bad_comps = [1 2 6 7];
            case 5
                bad_comps = [1 3 5];
            case 6
                bad_comps = [1 3 4 40];
            case 7
                bad_comps = [1 2 5 7];
            case 8
                bad_comps = [1 3 4 8 40];
            case 9
                bad_comps = [1 3 6 30];
        end
    % case 31 % sub-31
    %     switch run
    %         case 1
    %             bad_comps = [1 2 3 4 8 19];
    %         case 2
    %             bad_comps = [1 2 4 5 12];
    %         case 3
    %             bad_comps = [1 2 6 7 13];
    %         case 4
    %             bad_comps = [1 2 3 4 6];
    %         case 5
    %             bad_comps = [1 2 3 4 5 6];
    %         case 6
    %             bad_comps = [1 2 3 4 5 6];
    %         case 7
    %             bad_comps = [1 2 3 4 5];
    %         case 8
    %             bad_comps = [1 2 3 4 6];
    %     end
    case 31 % sub-31
        switch run
            case 1
                bad_comps = [1 2 3 5 11 21];
            case 3
                bad_comps = [1 2 3 4 6 12];
            case 4
                bad_comps = [1 2 3 6 7 15];
            case 5
                bad_comps = [1 2 3 4 5 12];
            case 6
                bad_comps = [1 2 3 4 5 13];
            case 7
                bad_comps = [1 2 3 4 5 6 17];
            case 8
                bad_comps = [1 2 3 4 10 11];
        end
    case 32 % sub-32
        switch run
            case 1
                bad_comps = [1 3 6 11 12 ];
            case 3
                bad_comps = [1 4 8 10 16 24];
            case 4
                bad_comps = [1 2 7 12 23];
            case 5
                bad_comps = [1 2 4 5 9 12];
            case 6
                bad_comps = [1 2 5 6 11 35];
            case 7
                bad_comps = [1 3 7 16 25];
            case 8
                bad_comps = [1 4 5 6 18 50];
            
        end
end

end