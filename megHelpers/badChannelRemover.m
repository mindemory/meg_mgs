function bad_chans = badChannelRemover(subjID, run)

% Reject components
switch subjID
    case 1 % sub-01
        bad_chans = [19 46]; % same for all 8 runs
    case 2 % sub-02
        switch run 
            case 1
                bad_chans = [10 19];
            case 2
                bad_chans = [19 27];
            case 3
                bad_chans = [];
            case 4
                bad_chans = [10 19 27 35 76 110];
            case 5
                bad_chans = [10 35 76 110];
            case 6
                bad_chans = [10 35 76 110];
            case 7
                bad_chans = [10 35 76 110];
            case 8
                bad_chans = [10 35 76 110];
        end

    case 3 % sub-03
        bad_chans = [];
    case 4 % sub-04
        if run == 4
            bad_chans = [74 120 125 128 136];
        elseif run == 6
            bad_chans = [27];
        elseif run == 8
            bad_chans = [74 120 125 128 136];
        elseif run == 9
            bad_chans = [19 27 35 74 120 125 128 136];
        else
            bad_chans = [];
        end
    case 5 % sub-05
        switch run
            case 1
                bad_chans = [4 9 10 11 12 13 14 18 20 25 28 34 35 51 58 67 ...
                            70 76 80 85 91 103 110 125 137 147 153 157];
            case 2
                bad_chans = [];
            case 3
                bad_chans = [];
            case 4
                bad_chans = [];
            case 5
                bad_chans = [];
            case 6
                bad_chans = [];
            case 7
                bad_chans = [];
            case 8
                bad_chans = [];
            case 9
                bad_chans = [];
            case 10
                bad_chans = [];
            case 11
                bad_chans = [];
            case 12
                bad_chans = [];
        end
    case 6 % sub-06
        if run == 3 || run == 5
            bad_chans = [136]; 
        else
            bad_chans = [];
        end
    case 7 % sub-07
        if run == 4 || run == 5
            bad_chans = [65];
        elseif run == 7 || run == 10 || run == 11
            bad_chans = [27];
        else
            bad_chans = [];
        end
    case 8 % sub-08
        bad_chans = [];
    case 9 % sub-09
        if run == 1
            bad_chans = [10];
        else
            bad_chans = [];
        end
    case 10 % sub-10
        if run == 3
            bad_chans = [4];
        elseif run == 7
            bad_chans = [7 10 11 12 14 16 18 20 25 34 36 39 41 59 67 70 76 ...
                        81 103 128 137 144 147 157];
        else
            bad_chans = [];
        end
    case 11 % sub-11
        if run == 1
            bad_chans = [4 11];
        elseif run == 5
            bad_chans = [4];
        else
            bad_chans = [];
        end
    case 12 % sub-12
        bad_chans = [4];
    case 13 % sub-13
        if run == 1
            bad_chans = [27];
        elseif run == 2 || run == 5 || run == 6 || run == 7
            bad_chans = [65 98];
        elseif run == 3
            bad_chans = [11 65 98 137];
        elseif run == 4
            bad_chans = [10 11 12 59 65];
        elseif run == 8
            bad_chans = [27 65 98];
        end
    case 15 % sub-15
        if run == 1 || run == 2
            bad_chans = [98];
        elseif run == 3
            bad_chans = [7 10 41 50 98];
        elseif run == 4 || run == 6 || run == 7 || run == 8
            bad_chans = [27 98];
        elseif run == 5
            bad_chans = [10 11 12 25 41 98];
        end
    case 16 % sub-16
        if run <= 3
            bad_chans = [27 98];
        elseif run == 4 || run == 5 || run == 6 || run == 7
            bad_chans = [98];
        elseif run == 8
            bad_chans = [27 98];
        end
    case 17 % sub-17
        if run == 1 || run == 2
            bad_chans = [65];
        elseif run == 3
            bad_chans = [27 65 77 112 135];
        elseif run >= 4
            bad_chans = [65 77 112 135];
        else
            bad_chans = [];
        end
    case 18 % sub-18
        if run <= 3 || run == 6 || run == 7 || run == 8
            bad_chans = [65];
        elseif run == 4 || run == 5
            bad_chans = [65 98];

        end
    case 19 % sub-19
        bad_chans = [];
    case 20 % sub-20
        if run <= 4
            bad_chans = [77 98 112 135];
        elseif run >= 5 
            bad_chans = [77 98 112 135 146];
        end

    case 21 % sub-21
        if run == 1
            bad_chans = [9 11 12 40 44 45 46 47];
        else
            bad_chans = [];
        end
    case 22 % sub-22
        bad_chans = [65];
    case 23 % sub-23
        if run == 6
            bad_chans = [10 11 25 35 49 50 58 65 98 103 110 125 157];
        else
            bad_chans = [65 98];
        end
    case 24 % sub-24
        bad_chans = [65 98];
    case 25 % sub-25
        bad_chans = [65 98];
    case 26 % sub-26
        if run <= 2
            bad_chans = [65];
        else
            bad_chans = [65 98];
        end
    case 27 % sub-27
        bad_chans = [65];
    case 28 % sub-28
        if run <= 3 || run == 5
            bad_chans = [110];
        else
            bad_chans = [46 110];
        end
    case 29 % sub-29
        bad_chans = [];
    case 30 % sub-30
        bad_chans = [];
    case 31 % sub-31
        bad_chans = [65];
        
    case 32 % sub-32
        bad_chans = [65];
end
end